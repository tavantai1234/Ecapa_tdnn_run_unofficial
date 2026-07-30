#!/usr/bin/env python3
"""Open-set speaker-verification evaluation for ECAPA-TDNN baseline E0.

The script performs the final evaluation stage:

    enrol/test recording
    -> split internally into chunks
    -> Fbank
    -> fine-tuned ECAPA-TDNN
    -> chunk embeddings
    -> mean recording embedding
    -> cosine similarity for verification trials
    -> EER and minDCF

The training classifier is loaded as part of the checkpoint, but it is NOT
used to score verification speakers.

Expected files
--------------
The script derives these paths from the directory containing train.csv:

    manifests/E0/enrol.csv
    manifests/E0/test.csv
    manifests/E0/verification_trials.txt

Results are written to:

    <output_folder>/verification/
        recording_embeddings.pt
        verification_scores.csv
        verification_metrics.json
        verification_metrics.txt

Usage
-----
    python test_e0.py hparams/baseline_mac.yaml

Force CPU:
    python test_e0.py hparams/baseline_mac.yaml --device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Unsupported MPS operations fall back to CPU instead of stopping.
# This must be set before importing torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import speechbrain as sb
import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio import audio_io
from speechbrain.utils.metric_stats import EER, minDCF


DEFAULT_CHUNK_SECONDS = 3.0
DEFAULT_MIN_CHUNK_SECONDS = 1.5
DEFAULT_EMBEDDING_BATCH_SIZE = 32

DEFAULT_P_TARGET = 0.01
DEFAULT_C_MISS = 1.0
DEFAULT_C_FA = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a fine-tuned ECAPA-TDNN checkpoint using "
            "enrol/test verification trials."
        )
    )
    parser.add_argument(
        "hparams_file",
        type=Path,
        help="Training HyperPyYAML file, e.g. hparams/baseline_mac.yaml",
    )
    parser.add_argument(
        "--device",
        default="mps",
        choices=["mps", "cpu", "cuda"],
        help="Evaluation device. Default: mps",
    )
    parser.add_argument(
        "--enrol-csv",
        type=Path,
        default=None,
        help="Optional enrol manifest override.",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=None,
        help="Optional test manifest override.",
    )
    parser.add_argument(
        "--trials",
        type=Path,
        default=None,
        help="Optional verification trials override.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional result directory override.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=DEFAULT_CHUNK_SECONDS,
        help="Internal evaluation chunk duration. Default: 3.0",
    )
    parser.add_argument(
        "--min-chunk-seconds",
        type=float,
        default=DEFAULT_MIN_CHUNK_SECONDS,
        help=(
            "Keep the final residual chunk when it is at least this long. "
            "Default: 1.5"
        ),
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        help="Number of chunks encoded together. Default: 32",
    )
    parser.add_argument(
        "--p-target",
        type=float,
        default=DEFAULT_P_TARGET,
        help="Target prior for minDCF. Default: 0.01",
    )
    parser.add_argument(
        "--c-miss",
        type=float,
        default=DEFAULT_C_MISS,
        help="Miss cost for minDCF. Default: 1.0",
    )
    parser.add_argument(
        "--c-fa",
        type=float,
        default=DEFAULT_C_FA,
        help="False-alarm cost for minDCF. Default: 1.0",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """Resolve a user/config path relative to the current working directory."""
    return path.expanduser().resolve()


def validate_args(args: argparse.Namespace) -> None:
    if args.chunk_seconds <= 0:
        raise ValueError("--chunk-seconds must be greater than 0.")

    if args.min_chunk_seconds <= 0:
        raise ValueError("--min-chunk-seconds must be greater than 0.")

    if args.min_chunk_seconds > args.chunk_seconds:
        raise ValueError(
            "--min-chunk-seconds cannot exceed --chunk-seconds."
        )

    if args.embedding_batch_size <= 0:
        raise ValueError("--embedding-batch-size must be greater than 0.")

    if not 0 < args.p_target < 1:
        raise ValueError("--p-target must be between 0 and 1.")

    if args.c_miss <= 0 or args.c_fa <= 0:
        raise ValueError("--c-miss and --c-fa must be greater than 0.")


def validate_device(device_name: str) -> torch.device:
    if device_name == "mps":
        if not torch.backends.mps.is_built():
            raise RuntimeError(
                "This PyTorch installation was not built with MPS support."
            )
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS is unavailable. Run with '--device cpu' or check "
                "the macOS/PyTorch installation."
            )

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run with '--device cpu' or '--device mps'."
        )

    return torch.device(device_name)


def load_hparams(hparams_file: Path):
    hparams_file = resolve_path(hparams_file)

    if not hparams_file.is_file():
        raise FileNotFoundError(
            f"Hyperparameter file does not exist: {hparams_file}"
        )

    with hparams_file.open(encoding="utf-8") as file:
        hparams = load_hyperpyyaml(file)

    return hparams_file, hparams


def derive_paths(
    hparams,
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path]:
    train_annotation = resolve_path(Path(hparams["train_annotation"]))
    manifest_dir = train_annotation.parent

    enrol_csv = resolve_path(
        args.enrol_csv
        if args.enrol_csv is not None
        else manifest_dir / "enrol.csv"
    )
    test_csv = resolve_path(
        args.test_csv
        if args.test_csv is not None
        else manifest_dir / "test.csv"
    )
    trials_path = resolve_path(
        args.trials
        if args.trials is not None
        else manifest_dir / "verification_trials.txt"
    )
    output_dir = resolve_path(
        args.output_dir
        if args.output_dir is not None
        else Path(hparams["output_folder"]) / "verification"
    )

    for label, path in (
        ("enrol manifest", enrol_csv),
        ("test manifest", test_csv),
        ("trials file", trials_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    return enrol_csv, test_csv, trials_path, output_dir


def recover_best_checkpoint(hparams):
    """Load the checkpoint with the lowest validation ErrorRate."""
    checkpoint = hparams["checkpointer"].recover_if_possible(
        min_key="ErrorRate"
    )

    if checkpoint is None:
        raise FileNotFoundError(
            "No training checkpoint was found in the configured save folder: "
            f"{hparams['save_folder']}"
        )

    return checkpoint


def prepare_modules(hparams, device: torch.device) -> dict[str, torch.nn.Module]:
    """Move only the modules required for embedding extraction."""
    modules = {
        "compute_features": hparams["compute_features"],
        "mean_var_norm": hparams["mean_var_norm"],
        "embedding_model": hparams["embedding_model"],
    }

    for module in modules.values():
        module.to(device)
        module.eval()

    return modules


def load_manifest(path: Path) -> dict[str, dict]:
    required_columns = {
        "ID",
        "duration",
        "wav",
        "start",
        "stop",
        "spk_id",
    }

    rows: dict[str, dict] = {}

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")

        missing = required_columns - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{path} is missing columns: {sorted(missing)}"
            )

        for line_number, row in enumerate(reader, start=2):
            recording_id = row["ID"].strip()

            if not recording_id:
                raise ValueError(
                    f"{path}:{line_number}: empty recording ID"
                )

            if recording_id in rows:
                raise ValueError(
                    f"{path}:{line_number}: duplicate ID {recording_id}"
                )

            start = int(row["start"])
            stop = int(row["stop"])

            if start < 0 or stop <= start:
                raise ValueError(
                    f"{path}:{line_number}: invalid start/stop "
                    f"for {recording_id}: {start}, {stop}"
                )

            wav_path = Path(row["wav"]).expanduser()
            if not wav_path.is_absolute():
                # prepare_e0.py normally writes absolute paths. This fallback
                # resolves relative paths from the current project directory.
                wav_path = wav_path.resolve()

            if not wav_path.is_file():
                raise FileNotFoundError(
                    f"{path}:{line_number}: audio not found: {wav_path}"
                )

            rows[recording_id] = {
                "ID": recording_id,
                "duration": float(row["duration"]),
                "wav": wav_path,
                "start": start,
                "stop": stop,
                "spk_id": row["spk_id"].strip(),
            }

    if not rows:
        raise ValueError(f"Manifest is empty: {path}")

    return rows


def load_trials(
    path: Path,
) -> list[tuple[int, str, str]]:
    trials: list[tuple[int, str, str]] = []

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) != 3:
                raise ValueError(
                    f"{path}:{line_number}: expected "
                    "'label enrol_id test_id'"
                )

            label_text, enrol_id, test_id = parts

            try:
                label = int(label_text)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: label must be 0 or 1"
                ) from exc

            if label not in (0, 1):
                raise ValueError(
                    f"{path}:{line_number}: label must be 0 or 1"
                )

            trials.append((label, enrol_id, test_id))

    if not trials:
        raise ValueError(f"No trials found in: {path}")

    if not any(label == 1 for label, _, _ in trials):
        raise ValueError("Trials contain no genuine pairs.")

    if not any(label == 0 for label, _, _ in trials):
        raise ValueError("Trials contain no impostor pairs.")

    return trials


def validate_trials(
    trials: list[tuple[int, str, str]],
    enrol_rows: dict[str, dict],
    test_rows: dict[str, dict],
) -> None:
    missing_enrol = sorted(
        {
            enrol_id
            for _, enrol_id, _ in trials
            if enrol_id not in enrol_rows
        }
    )
    missing_test = sorted(
        {
            test_id
            for _, _, test_id in trials
            if test_id not in test_rows
        }
    )

    if missing_enrol:
        raise ValueError(
            f"Trials reference unknown enrol IDs: {missing_enrol[:10]}"
        )

    if missing_test:
        raise ValueError(
            f"Trials reference unknown test IDs: {missing_test[:10]}"
        )

    inconsistent_labels = []

    for label, enrol_id, test_id in trials:
        expected_label = int(
            enrol_rows[enrol_id]["spk_id"]
            == test_rows[test_id]["spk_id"]
        )
        if label != expected_label:
            inconsistent_labels.append(
                (enrol_id, test_id, label, expected_label)
            )

    if inconsistent_labels:
        example = inconsistent_labels[0]
        raise ValueError(
            "Trial labels disagree with manifest speaker IDs. "
            f"Example: enrol={example[0]}, test={example[1]}, "
            f"label={example[2]}, expected={example[3]}"
        )


def load_audio_segment(
    row: dict,
    expected_sample_rate: int,
) -> torch.Tensor:
    expected_frames = row["stop"] - row["start"]

    signal, sample_rate = audio_io.load(
        str(row["wav"]),
        frame_offset=row["start"],
        num_frames=expected_frames,
    )

    if sample_rate != expected_sample_rate:
        raise ValueError(
            f"{row['wav']}: expected {expected_sample_rate} Hz, "
            f"got {sample_rate} Hz"
        )

    if signal.ndim == 1:
        mono = signal
    elif signal.ndim == 2 and signal.shape[0] == 1:
        mono = signal.squeeze(0)
    elif signal.ndim == 2 and signal.shape[1] == 1:
        mono = signal.squeeze(1)
    else:
        raise ValueError(
            f"{row['wav']}: expected mono audio, "
            f"got shape {tuple(signal.shape)}"
        )

    if mono.numel() != expected_frames:
        raise ValueError(
            f"{row['wav']}: requested {expected_frames} frames but "
            f"loaded {mono.numel()}. Check manifest start/stop."
        )

    if not torch.isfinite(mono).all():
        raise ValueError(f"{row['wav']}: audio contains NaN or infinity.")

    return mono.float().contiguous()


def split_waveform(
    waveform: torch.Tensor,
    chunk_frames: int,
    min_chunk_frames: int,
) -> list[torch.Tensor]:
    chunks: list[torch.Tensor] = []
    start = 0
    total_frames = waveform.numel()

    while start < total_frames:
        remaining = total_frames - start

        if remaining >= chunk_frames:
            stop = start + chunk_frames
        elif remaining >= min_chunk_frames:
            stop = total_frames
        else:
            break

        chunks.append(waveform[start:stop])
        start = stop

    if not chunks:
        raise ValueError(
            "Recording is too short to produce an evaluation chunk: "
            f"{total_frames} frames"
        )

    return chunks


@torch.inference_mode()
def encode_chunk_batch(
    chunks: list[torch.Tensor],
    modules: dict[str, torch.nn.Module],
    device: torch.device,
) -> torch.Tensor:
    actual_lengths = torch.tensor(
        [chunk.numel() for chunk in chunks],
        dtype=torch.float32,
    )
    max_length = int(actual_lengths.max().item())

    wavs = torch.zeros(
        len(chunks),
        max_length,
        dtype=torch.float32,
    )

    for index, chunk in enumerate(chunks):
        wavs[index, : chunk.numel()] = chunk

    relative_lengths = actual_lengths / max_length

    wavs = wavs.to(device)
    relative_lengths = relative_lengths.to(device)

    features = modules["compute_features"](wavs)
    features = modules["mean_var_norm"](
        features,
        relative_lengths,
    )
    embeddings = modules["embedding_model"](
        features,
        relative_lengths,
    )

    # ECAPA-TDNN normally returns [batch, 1, embedding_dim].
    if embeddings.ndim == 3 and embeddings.shape[1] == 1:
        embeddings = embeddings.squeeze(1)

    if embeddings.ndim != 2:
        raise ValueError(
            "Unexpected ECAPA embedding shape: "
            f"{tuple(embeddings.shape)}"
        )

    embeddings = F.normalize(embeddings, p=2, dim=-1)

    return embeddings.cpu()


@torch.inference_mode()
def create_recording_embedding(
    row: dict,
    modules: dict[str, torch.nn.Module],
    device: torch.device,
    sample_rate: int,
    chunk_seconds: float,
    min_chunk_seconds: float,
    embedding_batch_size: int,
) -> tuple[torch.Tensor, int]:
    waveform = load_audio_segment(
        row,
        expected_sample_rate=sample_rate,
    )

    chunk_frames = int(round(chunk_seconds * sample_rate))
    min_chunk_frames = int(round(min_chunk_seconds * sample_rate))

    chunks = split_waveform(
        waveform,
        chunk_frames=chunk_frames,
        min_chunk_frames=min_chunk_frames,
    )

    chunk_embeddings = []

    for start_index in range(0, len(chunks), embedding_batch_size):
        batch_chunks = chunks[
            start_index : start_index + embedding_batch_size
        ]
        batch_embeddings = encode_chunk_batch(
            batch_chunks,
            modules=modules,
            device=device,
        )
        chunk_embeddings.append(batch_embeddings)

    all_embeddings = torch.cat(chunk_embeddings, dim=0)

    # Average multiple chunk embeddings into one recording-level embedding.
    recording_embedding = all_embeddings.mean(dim=0)
    recording_embedding = F.normalize(
        recording_embedding,
        p=2,
        dim=0,
    )

    return recording_embedding, len(chunks)


def extract_embeddings(
    rows: dict[str, dict],
    split_name: str,
    modules: dict[str, torch.nn.Module],
    device: torch.device,
    sample_rate: int,
    chunk_seconds: float,
    min_chunk_seconds: float,
    embedding_batch_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    embeddings: dict[str, torch.Tensor] = {}
    chunk_counts: dict[str, int] = {}

    total = len(rows)

    for index, (recording_id, row) in enumerate(
        rows.items(),
        start=1,
    ):
        embedding, chunk_count = create_recording_embedding(
            row=row,
            modules=modules,
            device=device,
            sample_rate=sample_rate,
            chunk_seconds=chunk_seconds,
            min_chunk_seconds=min_chunk_seconds,
            embedding_batch_size=embedding_batch_size,
        )

        embeddings[recording_id] = embedding
        chunk_counts[recording_id] = chunk_count

        print(
            f"[{split_name} {index}/{total}] "
            f"{recording_id}: {chunk_count} chunk(s)"
        )

    return embeddings, chunk_counts


def score_trials(
    trials: list[tuple[int, str, str]],
    enrol_embeddings: dict[str, torch.Tensor],
    test_embeddings: dict[str, torch.Tensor],
) -> tuple[list[dict], torch.Tensor, torch.Tensor]:
    score_rows: list[dict] = []
    positive_scores: list[float] = []
    negative_scores: list[float] = []

    for label, enrol_id, test_id in trials:
        score = torch.dot(
            enrol_embeddings[enrol_id],
            test_embeddings[test_id],
        ).item()

        score_rows.append(
            {
                "label": label,
                "enrol_id": enrol_id,
                "test_id": test_id,
                "score": score,
            }
        )

        if label == 1:
            positive_scores.append(score)
        else:
            negative_scores.append(score)

    return (
        score_rows,
        torch.tensor(positive_scores, dtype=torch.float32),
        torch.tensor(negative_scores, dtype=torch.float32),
    )


def calculate_threshold_stats(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    threshold: float,
) -> dict[str, float | int]:
    false_rejects = int(
        (positive_scores < threshold).sum().item()
    )
    false_accepts = int(
        (negative_scores >= threshold).sum().item()
    )

    genuine_count = int(positive_scores.numel())
    impostor_count = int(negative_scores.numel())

    frr = false_rejects / genuine_count
    far = false_accepts / impostor_count

    return {
        "false_rejects": false_rejects,
        "false_accepts": false_accepts,
        "FRR": frr,
        "FAR": far,
    }


def write_scores(
    path: Path,
    score_rows: list[dict],
    threshold: float,
) -> None:
    columns = [
        "label",
        "enrol_id",
        "test_id",
        "score",
        "prediction_at_eer_threshold",
        "correct_at_eer_threshold",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for row in score_rows:
            prediction = int(row["score"] >= threshold)
            writer.writerow(
                {
                    **row,
                    "score": f"{row['score']:.10f}",
                    "prediction_at_eer_threshold": prediction,
                    "correct_at_eer_threshold": int(
                        prediction == row["label"]
                    ),
                }
            )


def checkpoint_description(checkpoint) -> str:
    for attribute in ("path", "name"):
        value = getattr(checkpoint, attribute, None)
        if value is not None:
            return str(value)
    return str(checkpoint)


def save_results(
    output_dir: Path,
    hparams_file: Path,
    checkpoint,
    enrol_embeddings: dict[str, torch.Tensor],
    test_embeddings: dict[str, torch.Tensor],
    enrol_chunk_counts: dict[str, int],
    test_chunk_counts: dict[str, int],
    score_rows: list[dict],
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    eer: float,
    eer_threshold: float,
    min_dcf: float,
    min_dcf_threshold: float,
    threshold_stats: dict,
    args: argparse.Namespace,
) -> dict:
    embeddings_path = output_dir / "recording_embeddings.pt"
    scores_path = output_dir / "verification_scores.csv"
    metrics_json_path = output_dir / "verification_metrics.json"
    metrics_txt_path = output_dir / "verification_metrics.txt"

    torch.save(
        {
            "enrol_embeddings": enrol_embeddings,
            "test_embeddings": test_embeddings,
            "enrol_chunk_counts": enrol_chunk_counts,
            "test_chunk_counts": test_chunk_counts,
        },
        embeddings_path,
    )

    write_scores(
        path=scores_path,
        score_rows=score_rows,
        threshold=eer_threshold,
    )

    metrics = {
        "hparams_file": str(hparams_file),
        "checkpoint": checkpoint_description(checkpoint),
        "chunk_seconds": args.chunk_seconds,
        "min_chunk_seconds": args.min_chunk_seconds,
        "embedding_batch_size": args.embedding_batch_size,
        "num_enrol_recordings": len(enrol_embeddings),
        "num_test_recordings": len(test_embeddings),
        "num_trials": len(score_rows),
        "num_genuine_trials": int(positive_scores.numel()),
        "num_impostor_trials": int(negative_scores.numel()),
        "EER": eer,
        "EER_percent": eer * 100.0,
        "EER_threshold": eer_threshold,
        "FAR_at_EER_threshold": threshold_stats["FAR"],
        "FRR_at_EER_threshold": threshold_stats["FRR"],
        "false_accepts_at_EER_threshold": (
            threshold_stats["false_accepts"]
        ),
        "false_rejects_at_EER_threshold": (
            threshold_stats["false_rejects"]
        ),
        "minDCF": min_dcf,
        "minDCF_threshold": min_dcf_threshold,
        "p_target": args.p_target,
        "c_miss": args.c_miss,
        "c_fa": args.c_fa,
        "scores_file": str(scores_path),
        "embeddings_file": str(embeddings_path),
    }

    with metrics_json_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    with metrics_txt_path.open("w", encoding="utf-8") as file:
        file.write("ECAPA-TDNN Open-Set Speaker Verification\n")
        file.write("========================================\n")
        file.write(f"Checkpoint: {metrics['checkpoint']}\n")
        file.write(
            f"Enrol recordings: {metrics['num_enrol_recordings']}\n"
        )
        file.write(
            f"Test recordings: {metrics['num_test_recordings']}\n"
        )
        file.write(f"Trials: {metrics['num_trials']}\n")
        file.write(
            f"Genuine trials: {metrics['num_genuine_trials']}\n"
        )
        file.write(
            f"Impostor trials: {metrics['num_impostor_trials']}\n"
        )
        file.write(f"EER: {metrics['EER_percent']:.4f}%\n")
        file.write(
            f"EER threshold: {metrics['EER_threshold']:.10f}\n"
        )
        file.write(
            f"FAR at EER threshold: "
            f"{metrics['FAR_at_EER_threshold']:.6f}\n"
        )
        file.write(
            f"FRR at EER threshold: "
            f"{metrics['FRR_at_EER_threshold']:.6f}\n"
        )
        file.write(f"minDCF: {metrics['minDCF']:.10f}\n")
        file.write(
            f"minDCF threshold: "
            f"{metrics['minDCF_threshold']:.10f}\n"
        )
        file.write(f"p_target: {metrics['p_target']}\n")
        file.write(f"c_miss: {metrics['c_miss']}\n")
        file.write(f"c_fa: {metrics['c_fa']}\n")

    return metrics


def main() -> int:
    args = parse_args()
    validate_args(args)

    device = validate_device(args.device)
    hparams_file, hparams = load_hparams(args.hparams_file)

    enrol_csv, test_csv, trials_path, output_dir = derive_paths(
        hparams,
        args,
    )

    print(f"Device:       {device}")
    print(f"Enrol CSV:    {enrol_csv}")
    print(f"Test CSV:     {test_csv}")
    print(f"Trials:       {trials_path}")
    print(f"Output:       {output_dir}")

    checkpoint = recover_best_checkpoint(hparams)
    print(f"Checkpoint:   {checkpoint_description(checkpoint)}")

    modules = prepare_modules(hparams, device)

    enrol_rows = load_manifest(enrol_csv)
    test_rows = load_manifest(test_csv)
    trials = load_trials(trials_path)

    validate_trials(
        trials,
        enrol_rows=enrol_rows,
        test_rows=test_rows,
    )

    sample_rate = int(hparams["sample_rate"])

    print("\nExtracting enrol embeddings...")
    enrol_embeddings, enrol_chunk_counts = extract_embeddings(
        rows=enrol_rows,
        split_name="enrol",
        modules=modules,
        device=device,
        sample_rate=sample_rate,
        chunk_seconds=args.chunk_seconds,
        min_chunk_seconds=args.min_chunk_seconds,
        embedding_batch_size=args.embedding_batch_size,
    )

    print("\nExtracting test embeddings...")
    test_embeddings, test_chunk_counts = extract_embeddings(
        rows=test_rows,
        split_name="test",
        modules=modules,
        device=device,
        sample_rate=sample_rate,
        chunk_seconds=args.chunk_seconds,
        min_chunk_seconds=args.min_chunk_seconds,
        embedding_batch_size=args.embedding_batch_size,
    )

    score_rows, positive_scores, negative_scores = score_trials(
        trials,
        enrol_embeddings=enrol_embeddings,
        test_embeddings=test_embeddings,
    )

    eer_value, eer_threshold = EER(
        positive_scores,
        negative_scores,
    )
    min_dcf_value, min_dcf_threshold = minDCF(
        positive_scores,
        negative_scores,
        c_miss=args.c_miss,
        c_fa=args.c_fa,
        p_target=args.p_target,
    )

    eer_value = float(eer_value)
    eer_threshold = float(eer_threshold)
    min_dcf_value = float(min_dcf_value)
    min_dcf_threshold = float(min_dcf_threshold)

    threshold_stats = calculate_threshold_stats(
        positive_scores,
        negative_scores,
        threshold=eer_threshold,
    )

    metrics = save_results(
        output_dir=output_dir,
        hparams_file=hparams_file,
        checkpoint=checkpoint,
        enrol_embeddings=enrol_embeddings,
        test_embeddings=test_embeddings,
        enrol_chunk_counts=enrol_chunk_counts,
        test_chunk_counts=test_chunk_counts,
        score_rows=score_rows,
        positive_scores=positive_scores,
        negative_scores=negative_scores,
        eer=eer_value,
        eer_threshold=eer_threshold,
        min_dcf=min_dcf_value,
        min_dcf_threshold=min_dcf_threshold,
        threshold_stats=threshold_stats,
        args=args,
    )

    print("\nVerification results")
    print("--------------------")
    print(f"Trials:        {metrics['num_trials']}")
    print(f"Genuine:       {metrics['num_genuine_trials']}")
    print(f"Impostor:      {metrics['num_impostor_trials']}")
    print(f"EER:           {metrics['EER_percent']:.4f}%")
    print(f"EER threshold: {metrics['EER_threshold']:.10f}")
    print(f"minDCF:        {metrics['minDCF']:.10f}")
    print(f"Saved to:      {output_dir}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nEvaluation interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"\nERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
