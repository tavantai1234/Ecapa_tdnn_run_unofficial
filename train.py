#!/usr/bin/env python3
"""Fine-tune ECAPA-TDNN with P x K batches and validation EER.

The training contract is read from hparams_validation_eer_friend.yaml. The
script supports CUDA AMP, two AdamW parameter groups, cosine scheduling per
successful optimizer update, fixed validation trials, EER early stopping, and
automatic recovery from last.pt.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import random
import sys
from collections import defaultdict
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path

# Harmless outside macOS. On Apple MPS, unsupported operations may fall back
# to CPU instead of stopping the whole training process.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import speechbrain as sb
import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio import audio_io
from speechbrain.utils.distributed import run_on_main
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

CHECKPOINT_SCHEMA = "ecapa_validation_eer_training"
CHECKPOINT_VERSION = 1
REQUIRED_MANIFEST_COLUMNS = {
    "ID",
    "duration",
    "wav",
    "start",
    "stop",
    "spk_id",
}
REQUIRED_TRIAL_COLUMNS = {
    "trial_id",
    "target",
    "left_id",
    "right_id",
    "left_spk_id",
    "right_spk_id",
}


def resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def read_manifest(path: str | Path) -> list[dict]:
    manifest_path = resolve_path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    rows: list[dict] = []
    seen_ids: set[str] = set()
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or ())
        missing = REQUIRED_MANIFEST_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"{manifest_path} is missing columns: {sorted(missing)}")

        for line_number, raw in enumerate(reader, start=2):
            utterance_id = raw["ID"].strip()
            speaker_id = raw["spk_id"].strip()
            wav = raw["wav"].strip()
            if not utterance_id or not speaker_id or not wav:
                raise ValueError(f"{manifest_path}:{line_number}: empty required value")
            if utterance_id in seen_ids:
                raise ValueError(
                    f"{manifest_path}:{line_number}: duplicate ID {utterance_id}"
                )
            seen_ids.add(utterance_id)

            start = int(raw["start"])
            stop = int(raw["stop"])
            duration = float(raw["duration"])
            if start < 0 or stop <= start or duration <= 0:
                raise ValueError(f"{manifest_path}:{line_number}: invalid time range")
            rows.append(
                {
                    "id": utterance_id,
                    "spk_id": speaker_id,
                    "wav": wav,
                    "start": start,
                    "stop": stop,
                    "duration": duration,
                }
            )

    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return rows


def read_validation_trials(
    path: str | Path,
    validation_rows: list[dict],
) -> list[dict]:
    trial_path = resolve_path(path)
    if not trial_path.is_file():
        raise FileNotFoundError(f"Validation trials do not exist: {trial_path}")

    ownership = {row["id"]: row["spk_id"] for row in validation_rows}
    trials: list[dict] = []
    seen_trial_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()

    with trial_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or ())
        missing = REQUIRED_TRIAL_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"{trial_path} is missing columns: {sorted(missing)}")

        for line_number, raw in enumerate(reader, start=2):
            trial_id = raw["trial_id"].strip()
            left_id = raw["left_id"].strip()
            right_id = raw["right_id"].strip()
            target = int(raw["target"])
            pair = tuple(sorted((left_id, right_id)))

            if target not in (0, 1):
                raise ValueError(f"{trial_path}:{line_number}: target must be 0 or 1")
            if trial_id in seen_trial_ids or pair in seen_pairs:
                raise ValueError(f"{trial_path}:{line_number}: duplicate trial or pair")
            if left_id not in ownership or right_id not in ownership:
                raise ValueError(f"{trial_path}:{line_number}: unknown validation ID")
            left_speaker = ownership[left_id]
            right_speaker = ownership[right_id]
            expected = int(left_speaker == right_speaker)
            if (
                raw["left_spk_id"].strip() != left_speaker
                or raw["right_spk_id"].strip() != right_speaker
                or target != expected
            ):
                raise ValueError(
                    f"{trial_path}:{line_number}: ownership/target mismatch"
                )

            seen_trial_ids.add(trial_id)
            seen_pairs.add(pair)
            trials.append(
                {
                    "trial_id": trial_id,
                    "target": target,
                    "left_id": left_id,
                    "right_id": right_id,
                }
            )

    target_values = {trial["target"] for trial in trials}
    if target_values != {0, 1}:
        raise ValueError("Validation trials must contain genuine and impostor pairs.")
    return trials


class AudioManifestDataset(Dataset):
    """Lazy audio dataset backed by a SpeechBrain-style CSV manifest."""

    def __init__(
        self,
        rows: list[dict],
        sample_rate: int,
        label_to_index: dict[str, int] | None = None,
    ) -> None:
        self.rows = rows
        self.sample_rate = sample_rate
        self.label_to_index = label_to_index

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        signal, sample_rate = audio_io.load(
            row["wav"],
            frame_offset=row["start"],
            num_frames=row["stop"] - row["start"],
        )
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"{row['wav']}: expected {self.sample_rate} Hz, got {sample_rate} Hz"
            )

        if signal.ndim == 2 and signal.shape[0] == 1:
            signal = signal.squeeze(0)
        elif signal.ndim == 2 and signal.shape[1] == 1:
            signal = signal.squeeze(1)
        if signal.ndim != 1:
            raise ValueError(
                f"{row['wav']}: expected mono audio, got shape {tuple(signal.shape)}"
            )

        item = {
            "id": row["id"],
            "spk_id": row["spk_id"],
            "signal": signal.float(),
        }
        if self.label_to_index is not None:
            try:
                item["label"] = self.label_to_index[row["spk_id"]]
            except KeyError as exc:
                raise ValueError(f"Unknown training speaker: {row['spk_id']}") from exc
        return item


def collate_audio(items: list[dict]) -> dict:
    if not items:
        raise ValueError("Cannot collate an empty batch.")
    signals = [item["signal"] for item in items]
    sample_lengths = torch.tensor(
        [signal.numel() for signal in signals],
        dtype=torch.float32,
    )
    padded = pad_sequence(signals, batch_first=True)
    relative_lengths = sample_lengths / padded.shape[1]

    batch = {
        "ids": [item["id"] for item in items],
        "spk_ids": [item["spk_id"] for item in items],
        "signals": padded,
        "lengths": relative_lengths,
    }
    if "label" in items[0]:
        batch["labels"] = torch.tensor(
            [item["label"] for item in items],
            dtype=torch.long,
        )
    return batch


def deterministic_epoch_seed(seed: int, epoch: int) -> int:
    value = hashlib.sha256(f"{seed}|{epoch}".encode()).digest()
    return int.from_bytes(value[:8], byteorder="big", signed=False)


class PKBatchSampler(Sampler[list[int]]):
    """Deterministic, exposure-balanced P-speaker/K-utterance sampler.

    A speaker is selected only from the least-exposed group. Within each
    speaker, rows are consumed from a shuffled queue before any row is reused.
    The K rounds are interleaved so every physical microbatch contains
    different speakers when ``P`` is divisible by the microbatch size.
    """

    def __init__(
        self,
        speaker_to_indices: dict[str, list[int]],
        speakers_per_batch: int,
        utterances_per_speaker: int,
        batches_per_epoch: int,
        seed: int,
        epoch: int,
        start_batch: int = 0,
    ) -> None:
        self.speaker_to_indices = speaker_to_indices
        self.speakers = sorted(speaker_to_indices)
        self.speakers_per_batch = speakers_per_batch
        self.utterances_per_speaker = utterances_per_speaker
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.epoch = epoch
        self.start_batch = start_batch

        if speakers_per_batch > len(self.speakers):
            raise ValueError(
                f"P={speakers_per_batch}, but train.csv contains only "
                f"{len(self.speakers)} speakers."
            )
        insufficient = [
            speaker
            for speaker, indices in speaker_to_indices.items()
            if len(indices) < utterances_per_speaker
        ]
        if insufficient:
            raise ValueError(
                f"K={utterances_per_speaker}, but these speakers have too "
                f"few train chunks: {insufficient[:10]}"
            )
        if not 0 <= start_batch <= batches_per_epoch:
            raise ValueError("Invalid start batch for P x K sampler.")

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(deterministic_epoch_seed(self.seed, self.epoch))
        queues = {
            speaker: list(indices)
            for speaker, indices in self.speaker_to_indices.items()
        }
        for queue in queues.values():
            rng.shuffle(queue)
        exposure = {speaker: 0 for speaker in self.speakers}

        for batch_index in range(self.batches_per_epoch):
            tie_break = {speaker: rng.random() for speaker in self.speakers}
            selected_speakers = sorted(
                self.speakers,
                key=lambda speaker: (exposure[speaker], tie_break[speaker]),
            )[: self.speakers_per_batch]

            selected_rows: dict[str, list[int]] = {}
            for speaker in selected_speakers:
                chosen: list[int] = []
                queue = queues[speaker]
                while len(chosen) < self.utterances_per_speaker:
                    if not queue:
                        queue.extend(self.speaker_to_indices[speaker])
                        rng.shuffle(queue)
                    candidate = queue.pop()
                    if candidate not in chosen:
                        chosen.append(candidate)
                selected_rows[speaker] = chosen
                exposure[speaker] += 1

            # Interleave K rounds. Therefore every physical slice of size 4
            # contains distinct speakers when P=16 and microbatch_size=4.
            batch = [
                selected_rows[speaker][utterance_index]
                for utterance_index in range(self.utterances_per_speaker)
                for speaker in selected_speakers
            ]
            if batch_index >= self.start_batch:
                yield batch

    def __len__(self) -> int:
        return self.batches_per_epoch - self.start_batch


def build_speaker_index(rows: list[dict]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        result[row["spk_id"]].append(index)
    return dict(result)


def squeeze_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    if embeddings.ndim == 3 and embeddings.shape[1] == 1:
        embeddings = embeddings.squeeze(1)
    if embeddings.ndim != 2:
        raise ValueError(f"Unexpected ECAPA embedding shape: {tuple(embeddings.shape)}")
    return embeddings


def create_grad_scaler(enabled: bool, initial_scale: float):
    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=enabled,
            init_scale=initial_scale,
        )
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(
            enabled=enabled,
            init_scale=initial_scale,
        )


def cosine_factor(
    completed_steps: int,
    total_steps: int,
    minimum: float,
) -> float:
    progress = min(max(completed_steps / total_steps, 0.0), 1.0)
    return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


def calculate_eer(
    scores: list[float],
    targets: list[int],
) -> dict[str, float]:
    """Calculate an O(N log N) interpolated EER and executable threshold.

    The early-stopping value matches the friend's linearly interpolated ROC
    crossing. The reported threshold is the closest empirical operating point
    and accepts a pair when ``score >= threshold``.
    """
    if len(scores) != len(targets) or not scores:
        raise ValueError("Scores and targets must have equal non-zero length.")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("Validation scores contain NaN or Inf.")
    if any(target not in (0, 1) for target in targets):
        raise ValueError("Validation targets must be 0 or 1.")

    positive_count = sum(targets)
    negative_count = len(targets) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Both genuine and impostor trials are required.")

    ordered = sorted(
        zip(scores, targets),
        key=lambda item: item[0],
        reverse=True,
    )
    points: list[tuple[float, float, float]] = [
        (math.nextafter(ordered[0][0], math.inf), 0.0, 1.0)
    ]
    accepted_positive = 0
    accepted_negative = 0
    position = 0
    while position < len(ordered):
        threshold = ordered[position][0]
        while position < len(ordered) and ordered[position][0] == threshold:
            if ordered[position][1] == 1:
                accepted_positive += 1
            else:
                accepted_negative += 1
            position += 1
        far = accepted_negative / negative_count
        frr = (positive_count - accepted_positive) / positive_count
        points.append((threshold, far, frr))

    crossing: tuple[float, float, float] | None = None
    for point in points:
        if point[1] == point[2]:
            crossing = point
            break
    if crossing is None:
        for first, second in zip(points, points[1:]):
            first_difference = first[1] - first[2]
            second_difference = second[1] - second[2]
            if first_difference < 0 < second_difference:
                weight = -first_difference / (
                    second_difference - first_difference
                )
                crossing = (
                    first[0] + weight * (second[0] - first[0]),
                    first[1] + weight * (second[1] - first[1]),
                    first[2] + weight * (second[2] - first[2]),
                )
                break
    if crossing is None:
        raise RuntimeError("Could not find the validation FAR/FRR crossing.")

    empirical = min(
        points[1:],
        key=lambda point: (
            abs(point[1] - point[2]),
            (point[1] + point[2]) / 2.0,
            -point[0],
        ),
    )
    interpolated_eer = (crossing[1] + crossing[2]) / 2.0
    return {
        "EER": interpolated_eer,
        "EER_percent": interpolated_eer * 100.0,
        "interpolated_threshold": crossing[0],
        "threshold": empirical[0],
        "FAR": empirical[1],
        "FRR": empirical[2],
    }


def calculate_threshold_stats(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    threshold: float,
) -> dict[str, float | int]:
    # Match the friend's executable convention: accept when score >= threshold.
    false_rejects = int((positive_scores < threshold).sum().item())
    false_accepts = int((negative_scores >= threshold).sum().item())
    genuine_count = int(positive_scores.numel())
    impostor_count = int(negative_scores.numel())
    frr = false_rejects / genuine_count
    far = false_accepts / impostor_count
    accuracy = 1.0 - (false_rejects + false_accepts) / (genuine_count + impostor_count)
    return {
        "FAR": far,
        "FRR": frr,
        "accuracy": accuracy,
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
    }


def validate_configuration(hparams: dict) -> None:
    required = {
        "seed",
        "sampler_seed",
        "train_annotation",
        "validation_annotation",
        "validation_trials",
        "output_folder",
        "out_n_neurons",
        "sample_rate",
        "speakers_per_batch",
        "utterances_per_speaker",
        "logical_batch_size",
        "microbatch_size",
        "gradient_accumulation",
        "ecapa_learning_rate",
        "aam_learning_rate",
        "weight_decay",
        "number_of_epochs",
        "updates_per_epoch",
        "max_scheduler_updates",
        "minimum_lr_factor",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "checkpoint_every_updates",
        "compute_features",
        "mean_var_norm",
        "embedding_model",
        "classifier",
        "compute_cost",
        "train_logger",
        "checkpointer",
        "pretrainer",
    }
    missing = required - set(hparams)
    if missing:
        raise ValueError(f"YAML is missing settings: {sorted(missing)}")

    logical = int(hparams["logical_batch_size"])
    p = int(hparams["speakers_per_batch"])
    k = int(hparams["utterances_per_speaker"])
    micro = int(hparams["microbatch_size"])
    accumulation = int(hparams["gradient_accumulation"])
    if min(p, k, logical, micro, accumulation) < 1:
        raise ValueError("P, K, batch, microbatch, and accumulation must be positive.")
    if p * k != logical:
        raise ValueError(
            "speakers_per_batch * utterances_per_speaker != logical_batch_size"
        )
    if micro * accumulation != logical:
        raise ValueError(
            "microbatch_size * gradient_accumulation != logical_batch_size"
        )
    if logical % micro:
        raise ValueError("logical_batch_size must be divisible by microbatch_size")
    if p % micro:
        raise ValueError(
            "speakers_per_batch must be divisible by microbatch_size so "
            "each physical microbatch contains distinct speakers"
        )
    if int(hparams["updates_per_epoch"]) < 1:
        raise ValueError("updates_per_epoch must be positive.")
    if int(hparams["max_scheduler_updates"]) < 1:
        raise ValueError("max_scheduler_updates must be positive.")
    expected_updates = int(hparams["number_of_epochs"]) * int(
        hparams["updates_per_epoch"]
    )
    if expected_updates != int(hparams["max_scheduler_updates"]):
        raise ValueError(
            "max_scheduler_updates must equal number_of_epochs * updates_per_epoch"
        )
    if not 0.0 <= float(hparams["minimum_lr_factor"]) <= 1.0:
        raise ValueError("minimum_lr_factor must be between 0 and 1.")
    if (
        min(
            float(hparams["ecapa_learning_rate"]),
            float(hparams["aam_learning_rate"]),
        )
        <= 0
    ):
        raise ValueError("Both learning rates must be positive.")
    if int(hparams["early_stopping_patience"]) < 1:
        raise ValueError("early_stopping_patience must be at least 1.")
    if float(hparams["early_stopping_min_delta"]) < 0:
        raise ValueError("early_stopping_min_delta cannot be negative.")
    if str(hparams.get("validation_metric", "EER")).upper() != "EER":
        raise ValueError("This train.py supports validation_metric: EER.")
    if int(hparams.get("warmup_updates", 0)) != 0:
        raise ValueError("This friend-matched setup requires zero warmup.")
    if bool(hparams.get("cache_fbank", False)):
        raise ValueError("cache_fbank=True is not implemented in this repository yet.")
    if int(hparams["checkpoint_every_updates"]) < 1:
        raise ValueError("checkpoint_every_updates must be positive.")
    if int(hparams.get("log_every_updates", 25)) < 1:
        raise ValueError("log_every_updates must be positive.")


def training_contract(
    hparams: dict,
    label_to_index: dict[str, int],
) -> dict:
    keys = (
        "sample_rate",
        "n_mels",
        "embedding_dim",
        "out_n_neurons",
        "aam_margin",
        "aam_scale",
        "speakers_per_batch",
        "utterances_per_speaker",
        "logical_batch_size",
        "microbatch_size",
        "gradient_accumulation",
        "ecapa_learning_rate",
        "aam_learning_rate",
        "weight_decay",
        "number_of_epochs",
        "updates_per_epoch",
        "max_scheduler_updates",
        "minimum_lr_factor",
        "sampler_seed",
    )
    return {
        "hparams": {key: hparams.get(key) for key in keys},
        "label_to_index": label_to_index,
        "input_sha256": {
            key: file_sha256(hparams[key])
            for key in (
                "train_annotation",
                "validation_annotation",
                "validation_trials",
            )
        },
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with resolve_path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ECAPATrainer:
    def __init__(
        self,
        hparams: dict,
        device: torch.device,
        train_rows: list[dict],
        validation_rows: list[dict],
        trials: list[dict],
        label_to_index: dict[str, int],
    ) -> None:
        self.hparams = hparams
        self.device = device
        self.train_rows = train_rows
        self.validation_rows = validation_rows
        self.trials = trials
        self.label_to_index = label_to_index
        self.speaker_to_indices = build_speaker_index(train_rows)

        self.compute_features = hparams["compute_features"].to(device)
        self.mean_var_norm = hparams["mean_var_norm"].to(device)
        self.embedding_model = hparams["embedding_model"].to(device)
        self.classifier = hparams["classifier"].to(device)
        self.compute_cost = hparams["compute_cost"]

        self.freeze_normalizer = bool(hparams.get("freeze_mean_var_norm", True))
        self.freeze_bn_stats = bool(hparams.get("freeze_batchnorm_running_stats", True))
        self.train_bn_affine = bool(hparams.get("train_batchnorm_affine", True))
        if bool(hparams.get("freeze_embedding_model", False)):
            for parameter in self.embedding_model.parameters():
                parameter.requires_grad = False
        if self.freeze_normalizer:
            for parameter in self.mean_var_norm.parameters():
                parameter.requires_grad = False

        ecapa_parameters = [
            parameter
            for parameter in self.embedding_model.parameters()
            if parameter.requires_grad
        ]
        aam_parameters = [
            parameter
            for parameter in self.classifier.parameters()
            if parameter.requires_grad
        ]
        if not ecapa_parameters or not aam_parameters:
            raise ValueError(
                "The friend-matched setup requires trainable ECAPA and "
                "AAM parameter groups."
            )

        self.optimizer = torch.optim.AdamW(
            [
                {
                    "name": "ecapa",
                    "params": ecapa_parameters,
                    "lr": float(hparams["ecapa_learning_rate"]),
                },
                {
                    "name": "aam",
                    "params": aam_parameters,
                    "lr": float(hparams["aam_learning_rate"]),
                },
            ],
            weight_decay=float(hparams["weight_decay"]),
        )

        total_steps = int(hparams["max_scheduler_updates"])
        minimum_factor = float(hparams["minimum_lr_factor"])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_factor(
                step,
                total_steps,
                minimum_factor,
            ),
        )

        requested_amp = bool(hparams.get("use_amp", True))
        self.amp_enabled = requested_amp and device.type == "cuda"
        if requested_amp and not self.amp_enabled:
            print(
                "AMP requested but CUDA is not active; training in FP32 "
                f"on {device.type}."
            )
        self.scaler = create_grad_scaler(
            self.amp_enabled,
            float(hparams.get("grad_scaler_initial_scale", 128.0)),
        )

        self.output_folder = resolve_path(hparams["output_folder"])
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.last_checkpoint = self.output_folder / str(
            hparams.get("last_checkpoint_name", "last.pt")
        )
        self.best_checkpoint = self.output_folder / str(
            hparams.get("best_checkpoint_name", "best.pt")
        )

        self.next_epoch = 0
        self.next_batch_position = 0
        self.global_step = 0
        self.best_eer: float | None = None
        self.best_epoch: int | None = None
        self.best_threshold: float | None = None
        self.patience_counter = 0
        self.stopped_early = False

    def encoder_autocast(self):
        if not self.amp_enabled:
            return nullcontext()
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )

    def set_train_mode(self) -> None:
        self.compute_features.eval()
        if self.freeze_normalizer:
            self.mean_var_norm.eval()
        else:
            self.mean_var_norm.train()
        self.embedding_model.train()
        self.classifier.train()

        if self.freeze_bn_stats:
            for module in self.embedding_model.modules():
                if isinstance(module, _BatchNorm):
                    module.eval()
                    if module.affine:
                        module.weight.requires_grad = self.train_bn_affine
                        module.bias.requires_grad = self.train_bn_affine

    def set_evaluation_mode(self) -> None:
        self.compute_features.eval()
        self.mean_var_norm.eval()
        self.embedding_model.eval()
        self.classifier.eval()

    def encode(
        self,
        signals: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        features = self.compute_features(signals)
        features = self.mean_var_norm(features, lengths)
        with self.encoder_autocast():
            embeddings = self.embedding_model(features, lengths)
        embeddings = squeeze_embeddings(embeddings.float())
        if not bool(torch.isfinite(embeddings).all().item()):
            raise RuntimeError("ECAPA produced a NaN or Inf embedding.")
        return embeddings

    def train_logical_batch(self, batch: dict) -> float:
        signals = batch["signals"].to(self.device, non_blocking=True)
        lengths = batch["lengths"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        microbatch_size = int(self.hparams["microbatch_size"])
        accumulation = int(self.hparams["gradient_accumulation"])
        expected = microbatch_size * accumulation
        if signals.shape[0] != expected:
            raise ValueError(
                f"Expected logical batch {expected}, got {signals.shape[0]}"
            )

        # Retry once after an AMP overflow, matching the reference behavior.
        for attempt in range(2):
            self.optimizer.zero_grad(set_to_none=True)
            logical_loss = 0.0
            for start in range(0, expected, microbatch_size):
                stop = start + microbatch_size
                embeddings = self.encode(
                    signals[start:stop],
                    lengths[start:stop],
                )
                # Classifier and AAM loss deliberately remain FP32.
                predictions = self.classifier(embeddings.float())
                raw_loss = self.compute_cost(
                    predictions,
                    labels[start:stop].unsqueeze(1),
                    lengths[start:stop],
                )
                if raw_loss.ndim:
                    raw_loss = raw_loss.mean()
                if not bool(torch.isfinite(raw_loss).item()):
                    raise RuntimeError("AAM-Softmax produced a NaN or Inf loss.")
                logical_loss += float(raw_loss.detach().item()) / accumulation
                scaled_loss = raw_loss / accumulation
                if self.amp_enabled:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

            gradient_clip = self.hparams.get("gradient_clipping")
            if self.amp_enabled:
                self.scaler.unscale_(self.optimizer)
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for group in self.optimizer.param_groups
                        for parameter in group["params"]
                    ],
                    float(gradient_clip),
                )

            if self.amp_enabled:
                scale_before = float(self.scaler.get_scale())
                self.scaler.step(self.optimizer)
                self.scaler.update()
                overflow = float(self.scaler.get_scale()) < scale_before
                if overflow:
                    print(
                        f"AMP overflow at update {self.global_step + 1}; "
                        f"retry {attempt + 1}/2."
                    )
                    continue
            else:
                self.optimizer.step()

            self.scheduler.step()
            self.global_step += 1
            return logical_loss

        raise RuntimeError("AMP overflow persisted after one retry.")

    def make_train_loader(
        self,
        epoch: int,
        start_batch: int,
    ) -> DataLoader:
        dataset = AudioManifestDataset(
            self.train_rows,
            int(self.hparams["sample_rate"]),
            self.label_to_index,
        )
        sampler = PKBatchSampler(
            self.speaker_to_indices,
            int(self.hparams["speakers_per_batch"]),
            int(self.hparams["utterances_per_speaker"]),
            int(self.hparams["updates_per_epoch"]),
            int(self.hparams.get("sampler_seed", self.hparams["seed"])),
            epoch,
            start_batch,
        )
        options = self.hparams.get("train_dataloader_options", {})
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_audio,
            num_workers=int(options.get("num_workers", 0)),
            pin_memory=bool(options.get("pin_memory", False))
            and self.device.type == "cuda",
        )

    def make_validation_loader(self) -> DataLoader:
        dataset = AudioManifestDataset(
            self.validation_rows,
            int(self.hparams["sample_rate"]),
        )
        options = self.hparams.get(
            "validation_dataloader_options",
            {},
        )
        return DataLoader(
            dataset,
            batch_size=int(options.get("batch_size", 64)),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_audio,
            num_workers=int(options.get("num_workers", 0)),
            pin_memory=bool(options.get("pin_memory", False))
            and self.device.type == "cuda",
        )

    @torch.inference_mode()
    def validate(self) -> dict[str, float | int]:
        self.set_evaluation_mode()
        embeddings_by_id: dict[str, torch.Tensor] = {}
        for batch in self.make_validation_loader():
            signals = batch["signals"].to(
                self.device,
                non_blocking=True,
            )
            lengths = batch["lengths"].to(
                self.device,
                non_blocking=True,
            )
            embeddings = F.normalize(
                self.encode(signals, lengths),
                p=2,
                dim=-1,
            ).cpu()
            for utterance_id, embedding in zip(
                batch["ids"],
                embeddings,
            ):
                embeddings_by_id[utterance_id] = embedding

        positive_scores: list[float] = []
        negative_scores: list[float] = []
        for trial in self.trials:
            score = float(
                torch.dot(
                    embeddings_by_id[trial["left_id"]],
                    embeddings_by_id[trial["right_id"]],
                ).item()
            )
            if trial["target"] == 1:
                positive_scores.append(score)
            else:
                negative_scores.append(score)

        scores = positive_scores + negative_scores
        targets = [1] * len(positive_scores) + [0] * len(negative_scores)
        eer_stats = calculate_eer(scores, targets)
        positives = torch.tensor(positive_scores, dtype=torch.float32)
        negatives = torch.tensor(negative_scores, dtype=torch.float32)
        threshold = float(eer_stats["threshold"])
        stats = calculate_threshold_stats(
            positives,
            negatives,
            threshold,
        )
        return {
            **eer_stats,
            **stats,
            "genuine_trials": len(positive_scores),
            "impostor_trials": len(negative_scores),
        }

    def checkpoint_state(
        self,
        reason: str,
        stopped_early: bool | None = None,
    ) -> dict:
        state = {
            "schema": CHECKPOINT_SCHEMA,
            "version": CHECKPOINT_VERSION,
            "reason": reason,
            "contract": training_contract(
                self.hparams,
                self.label_to_index,
            ),
            "embedding_model_state_dict": (self.embedding_model.state_dict()),
            "classifier_state_dict": self.classifier.state_dict(),
            "mean_var_norm_state_dict": self.mean_var_norm.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "grad_scaler_state_dict": self.scaler.state_dict(),
            "next_epoch": self.next_epoch,
            "next_batch_position": self.next_batch_position,
            "global_step": self.global_step,
            "best_eer": self.best_eer,
            "best_epoch": self.best_epoch,
            "best_threshold": self.best_threshold,
            "patience_counter": self.patience_counter,
            "stopped_early": (
                self.stopped_early if stopped_early is None else stopped_early
            ),
            "python_random_state": random.getstate(),
            "torch_random_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda_random_states"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def atomic_torch_save(state: dict, path: Path) -> None:
        temporary = path.with_name(f"{path.name}.tmp")
        torch.save(state, temporary)
        os.replace(temporary, path)

    def save_last(self, reason: str) -> None:
        self.atomic_torch_save(
            self.checkpoint_state(reason),
            self.last_checkpoint,
        )

    def save_best(self, reason: str) -> None:
        self.atomic_torch_save(
            self.checkpoint_state(reason, stopped_early=False),
            self.best_checkpoint,
        )

    def load_checkpoint(self, path: Path) -> None:
        try:
            checkpoint = torch.load(
                path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)

        if (
            checkpoint.get("schema") != CHECKPOINT_SCHEMA
            or checkpoint.get("version") != CHECKPOINT_VERSION
        ):
            raise ValueError(f"Unsupported training checkpoint: {path}")
        expected_contract = training_contract(
            self.hparams,
            self.label_to_index,
        )
        if checkpoint.get("contract") != expected_contract:
            raise ValueError(
                "Checkpoint and current YAML/train manifest disagree. "
                "Use the original configuration or start a new output folder."
            )

        self.embedding_model.load_state_dict(checkpoint["embedding_model_state_dict"])
        self.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        self.mean_var_norm.load_state_dict(checkpoint["mean_var_norm_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.amp_enabled and checkpoint["grad_scaler_state_dict"]:
            self.scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])

        self.next_epoch = int(checkpoint["next_epoch"])
        self.next_batch_position = int(checkpoint["next_batch_position"])
        self.global_step = int(checkpoint["global_step"])
        self.best_eer = checkpoint["best_eer"]
        self.best_epoch = checkpoint["best_epoch"]
        self.best_threshold = checkpoint["best_threshold"]
        self.patience_counter = int(checkpoint["patience_counter"])
        self.stopped_early = bool(checkpoint.get("stopped_early", False))

        max_epochs = int(self.hparams["number_of_epochs"])
        batches_per_epoch = int(self.hparams["updates_per_epoch"])
        if not 0 <= self.next_epoch <= max_epochs:
            raise ValueError("Checkpoint epoch cursor is outside the schedule.")
        if not 0 <= self.next_batch_position <= batches_per_epoch:
            raise ValueError("Checkpoint batch cursor is outside the epoch.")
        if self.next_epoch == max_epochs and self.next_batch_position != 0:
            raise ValueError("A completed checkpoint must have batch cursor zero.")
        expected_step = (
            self.next_epoch * batches_per_epoch
            + self.next_batch_position
        )
        if self.global_step != expected_step:
            raise ValueError(
                "Checkpoint optimizer step disagrees with its epoch/batch cursor."
            )
        if self.global_step > int(self.hparams["max_scheduler_updates"]):
            raise ValueError("Checkpoint exceeds the cosine schedule horizon.")
        scheduler_epoch = int(self.scheduler.state_dict()["last_epoch"])
        if scheduler_epoch != self.global_step:
            raise ValueError(
                "Checkpoint scheduler position disagrees with optimizer steps."
            )
        if not 0 <= self.patience_counter <= int(
            self.hparams["early_stopping_patience"]
        ):
            raise ValueError("Checkpoint patience counter is invalid.")

        random.setstate(checkpoint["python_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"].cpu())
        if torch.cuda.is_available() and "cuda_random_states" in checkpoint:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_random_states"]]
            )
        print(
            f"Resumed from {path}: epoch={self.next_epoch + 1}, "
            f"batch={self.next_batch_position}, "
            f"update={self.global_step}"
        )

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        validation_stats: dict,
    ) -> None:
        learning_rates = {
            group.get("name", f"group_{index}"): group["lr"]
            for index, group in enumerate(self.optimizer.param_groups)
        }
        self.hparams["train_logger"].log_stats(
            stats_meta={
                "epoch": epoch + 1,
                "global_step": self.global_step,
                "ecapa_lr": learning_rates.get("ecapa"),
                "aam_lr": learning_rates.get("aam"),
            },
            train_stats={"loss": train_loss},
            valid_stats=validation_stats,
        )

    def save_speechbrain_best(
        self,
        validation_stats: dict,
    ) -> None:
        # ErrorRate is duplicated for backward compatibility with test_e0.py,
        # whose existing loader asks the checkpointer for min_key=ErrorRate.
        eer = float(validation_stats["EER"])
        self.hparams["checkpointer"].save_and_keep_only(
            meta={
                "EER": eer,
                "ErrorRate": eer,
                "threshold": float(validation_stats["threshold"]),
                "global_step": self.global_step,
            },
            min_keys=["EER"],
        )

    def fit(self) -> None:
        max_epochs = int(self.hparams["number_of_epochs"])
        batches_per_epoch = int(self.hparams["updates_per_epoch"])
        checkpoint_interval = int(self.hparams["checkpoint_every_updates"])
        patience = int(self.hparams["early_stopping_patience"])
        min_delta = float(self.hparams["early_stopping_min_delta"])
        log_every = int(self.hparams.get("log_every_updates", 25))

        if self.stopped_early:
            print(
                "last.pt records a completed early-stopped run. Set "
                "auto_resume=False or choose another output_folder to start "
                "again."
            )
            return
        if self.next_epoch >= max_epochs:
            print("Training is already complete according to last.pt.")
            return

        try:
            for epoch in range(self.next_epoch, max_epochs):
                start_batch = (
                    self.next_batch_position if epoch == self.next_epoch else 0
                )
                self.next_epoch = epoch
                self.next_batch_position = start_batch
                self.set_train_mode()
                loader = self.make_train_loader(epoch, start_batch)

                loss_sum = 0.0
                processed = 0
                for batch_index, batch in enumerate(
                    loader,
                    start=start_batch,
                ):
                    self.next_batch_position = batch_index
                    loss = self.train_logical_batch(batch)
                    loss_sum += loss
                    processed += 1
                    self.next_batch_position = batch_index + 1

                    if self.global_step % log_every == 0:
                        print(
                            f"Epoch {epoch + 1}/{max_epochs} | "
                            f"batch {batch_index + 1}/{batches_per_epoch} | "
                            f"update {self.global_step} | "
                            f"loss {loss:.6f}"
                        )
                    if self.global_step % checkpoint_interval == 0:
                        self.save_last("periodic_update")

                if processed == 0 and start_batch < batches_per_epoch:
                    raise RuntimeError("Training loader produced no batches.")
                train_loss = loss_sum / max(processed, 1)
                validation_stats = self.validate()

                current_eer = float(validation_stats["EER"])
                improved = (
                    self.best_eer is None or current_eer < self.best_eer - min_delta
                )
                if improved:
                    self.best_eer = current_eer
                    self.best_epoch = epoch + 1
                    self.best_threshold = float(validation_stats["threshold"])
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                self.next_epoch = epoch + 1
                self.next_batch_position = 0
                self.log_epoch(epoch, train_loss, validation_stats)

                print(
                    f"Validation epoch {epoch + 1}: "
                    f"EER={validation_stats['EER_percent']:.4f}% | "
                    f"threshold={validation_stats['threshold']:.6f} | "
                    f"FAR={validation_stats['FAR']:.6f} | "
                    f"FRR={validation_stats['FRR']:.6f} | "
                    f"patience={self.patience_counter}/{patience}"
                )

                if improved:
                    self.save_best("best_validation_eer")
                    self.save_speechbrain_best(validation_stats)

                self.stopped_early = self.patience_counter >= patience
                self.save_last("epoch_complete")
                if self.stopped_early:
                    print(
                        f"Early stopping at epoch {epoch + 1}. Best EER "
                        f"was {self.best_eer * 100.0:.4f}% at epoch "
                        f"{self.best_epoch}."
                    )
                    break
        except KeyboardInterrupt:
            self.save_last("keyboard_interrupt")
            print(f"\nTraining interrupted. State was saved to {self.last_checkpoint}")
            raise


def prepare_data(hparams: dict):
    train_rows = read_manifest(hparams["train_annotation"])
    validation_rows = read_manifest(hparams["validation_annotation"])

    train_speakers = sorted({row["spk_id"] for row in train_rows})
    validation_speakers = {row["spk_id"] for row in validation_rows}
    overlap = sorted(set(train_speakers) & validation_speakers)
    if overlap:
        raise ValueError(
            f"Speaker leakage between train and validation: {overlap[:10]}"
        )
    if len(train_speakers) != int(hparams["out_n_neurons"]):
        raise ValueError(
            f"train.csv has {len(train_speakers)} speakers, but "
            f"out_n_neurons={hparams['out_n_neurons']} in YAML."
        )

    label_to_index = {speaker: index for index, speaker in enumerate(train_speakers)}
    trials = read_validation_trials(
        hparams["validation_trials"],
        validation_rows,
    )
    return (
        train_rows,
        validation_rows,
        trials,
        label_to_index,
    )


def best_available_device() -> str:
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_default_device() -> None:
    if not any(
        argument == "--device" or argument.startswith("--device=")
        for argument in sys.argv[1:]
    ):
        sys.argv.append(f"--device={best_available_device()}")


def validate_device(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if device.startswith("mps"):
        backend = getattr(torch.backends, "mps", None)
        if backend is None or not backend.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")


def resolve_resume_checkpoint(hparams: dict) -> Path | None:
    explicit = hparams.get("resume_checkpoint")
    if explicit:
        path = resolve_path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        return path
    if not bool(hparams.get("auto_resume", True)):
        return None
    last_path = resolve_path(hparams["output_folder"]) / str(
        hparams.get("last_checkpoint_name", "last.pt")
    )
    return last_path if last_path.is_file() else None


def load_pretrained_if_needed(
    hparams: dict,
    resume_path: Path | None,
) -> None:
    if resume_path is not None:
        print("Resume checkpoint found; pretrained initialization is skipped.")
        return
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected()


def main() -> int:
    set_default_device()
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    device_name = str(run_opts["device"])
    validate_device(device_name)
    device = torch.device(device_name)
    print(f"Selected device: {device}")
    if device.type == "cuda":
        print(f"CUDA GPU: {torch.cuda.get_device_name(device)}")

    with open(hparams_file, encoding="utf-8") as stream:
        hparams = load_hyperpyyaml(stream, overrides)
    validate_configuration(hparams)

    os.makedirs(hparams["save_folder"], exist_ok=True)
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    (
        train_rows,
        validation_rows,
        trials,
        label_to_index,
    ) = prepare_data(hparams)
    resume_path = resolve_resume_checkpoint(hparams)
    load_pretrained_if_needed(hparams, resume_path)

    trainer = ECAPATrainer(
        hparams=hparams,
        device=device,
        train_rows=train_rows,
        validation_rows=validation_rows,
        trials=trials,
        label_to_index=label_to_index,
    )
    if resume_path is not None:
        trainer.load_checkpoint(resume_path)

    print(
        f"Train speakers: {len(label_to_index)} | "
        f"train chunks: {len(train_rows)} | "
        f"validation recordings: {len(validation_rows)} | "
        f"validation trials: {len(trials)}"
    )
    trainer.fit()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"\nERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
