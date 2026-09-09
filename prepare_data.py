#!/usr/bin/env python3
"""Create manifests for ECAPA-TDNN classification and verification.

Examples
--------
Legacy classification train/dev manifests (same speakers in both splits):

    python prepare_data.py train \
        --data-root processed/train_val \
        --output-dir manifests/train_val

Speaker-disjoint train/validation manifests for validation EER:

    python prepare_data.py train-eer \
        --data-root processed/train_val \
        --output-dir manifests/train_val_eer \
        --validation-speakers 61 \
        --validation-genuine-trials 10000 \
        --validation-impostor-trials 10000 \
        --seed 2026 \
        --trial-seed 2026

External verification manifests:

    python prepare_data.py test \
        --data-root processed/external_test \
        --output-dir manifests/external_test

Each top-level directory below --data-root is treated as one speaker.
Input recordings must already be mono and have the requested sample rate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import random
import re
import sys
from collections import Counter
from pathlib import Path

import soundfile as sf


DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_SEED = 42
DEFAULT_DEV_RATIO = 0.10
DEFAULT_CHUNK_SECONDS = 3.0
DEFAULT_MIN_CHUNK_SECONDS = 1.5
DEFAULT_VALIDATION_SPEAKER_RATIO = 0.10
DEFAULT_VALIDATION_GENUINE_TRIALS = 10_000
DEFAULT_VALIDATION_IMPOSTOR_TRIALS = 10_000
DEFAULT_TRIAL_SEED = 2026
DEFAULT_ENROL_COUNT = 5
DEFAULT_MIN_VERIFICATION_SECONDS = 1.5
AUDIO_EXTENSIONS = {".wav", ".flac"}

MANIFEST_COLUMNS = ["ID", "duration", "wav", "start", "stop", "spk_id"]
AUDIT_COLUMNS = [
    "recording_id",
    "relative_path",
    "spk_id",
    "split",
    "duration",
    "frames",
]
SPEAKER_SPLIT_COLUMNS = ["spk_id", "split", "ranking_sha256"]
VALIDATION_TRIAL_COLUMNS = [
    "trial_id",
    "target",
    "left_id",
    "right_id",
    "left_spk_id",
    "right_spk_id",
]


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Processed data root; each top-level directory is one speaker.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which manifest files are written.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite only manifest files produced by this command.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare classification, validation-EER, or external-test "
            "manifests."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser(
        "train",
        help="Create train.csv and dev.csv.",
    )
    add_common_arguments(train_parser)
    train_parser.add_argument(
        "--dev-ratio",
        type=float,
        default=DEFAULT_DEV_RATIO,
    )
    train_parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=DEFAULT_CHUNK_SECONDS,
    )
    train_parser.add_argument(
        "--min-chunk-seconds",
        type=float,
        default=DEFAULT_MIN_CHUNK_SECONDS,
    )

    train_eer_parser = commands.add_parser(
        "train-eer",
        help=(
            "Create speaker-disjoint train.csv, validation.csv, and "
            "validation_trials.csv."
        ),
    )
    add_common_arguments(train_eer_parser)
    validation_size = train_eer_parser.add_mutually_exclusive_group()
    validation_size.add_argument(
        "--validation-speakers",
        type=int,
        default=None,
        help=(
            "Exact number of validation speakers. For the friend's "
            "488/61 train/validation setup, use 61."
        ),
    )
    validation_size.add_argument(
        "--validation-speaker-ratio",
        type=float,
        default=None,
        help=(
            "Validation speaker ratio when --validation-speakers is not "
            "provided (default: 0.10)."
        ),
    )
    train_eer_parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=DEFAULT_CHUNK_SECONDS,
    )
    train_eer_parser.add_argument(
        "--min-chunk-seconds",
        type=float,
        default=DEFAULT_MIN_CHUNK_SECONDS,
    )
    train_eer_parser.add_argument(
        "--min-validation-seconds",
        type=float,
        default=DEFAULT_MIN_VERIFICATION_SECONDS,
        help="Minimum duration of a recording used in validation trials.",
    )
    train_eer_parser.add_argument(
        "--validation-genuine-trials",
        type=int,
        default=DEFAULT_VALIDATION_GENUINE_TRIALS,
    )
    train_eer_parser.add_argument(
        "--validation-impostor-trials",
        type=int,
        default=DEFAULT_VALIDATION_IMPOSTOR_TRIALS,
    )
    train_eer_parser.add_argument(
        "--trial-seed",
        type=int,
        default=DEFAULT_TRIAL_SEED,
        help="Separate seed used only for fixed validation-trial sampling.",
    )

    test_parser = commands.add_parser(
        "test",
        help="Create enrol.csv, test.csv, and verification_trials.txt.",
    )
    add_common_arguments(test_parser)
    test_parser.add_argument(
        "--enrol-recordings-per-speaker",
        type=int,
        default=DEFAULT_ENROL_COUNT,
    )
    test_parser.add_argument(
        "--min-verification-seconds",
        type=float,
        default=DEFAULT_MIN_VERIFICATION_SECONDS,
    )
    test_parser.add_argument(
        "--all-impostor-trials",
        action="store_true",
        help="Keep all different-speaker trials instead of balancing them.",
    )
    test_parser.add_argument(
        "--train-csv",
        type=Path,
        default=None,
        help="Optional train.csv used to check speaker disjointness.",
    )
    test_parser.add_argument(
        "--validation-csv",
        type=Path,
        default=None,
        help="Optional validation.csv used to check speaker disjointness.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be greater than zero.")

    if args.command in {"train", "train-eer"}:
        if args.chunk_seconds <= 0 or args.min_chunk_seconds <= 0:
            raise ValueError("Chunk durations must be greater than zero.")
        if args.min_chunk_seconds > args.chunk_seconds:
            raise ValueError(
                "--min-chunk-seconds cannot exceed --chunk-seconds."
            )

    if args.command == "train":
        if not 0.0 < args.dev_ratio < 1.0:
            raise ValueError("--dev-ratio must be between 0 and 1.")

    elif args.command == "train-eer":
        ratio = args.validation_speaker_ratio
        if ratio is not None and not 0.0 < ratio < 1.0:
            raise ValueError(
                "--validation-speaker-ratio must be between 0 and 1."
            )
        if args.validation_speakers is not None:
            if args.validation_speakers < 2:
                raise ValueError("--validation-speakers must be at least 2.")
        if args.min_validation_seconds <= 0:
            raise ValueError(
                "--min-validation-seconds must be greater than zero."
            )
        if args.validation_genuine_trials < 1:
            raise ValueError(
                "--validation-genuine-trials must be at least 1."
            )
        if args.validation_impostor_trials < 1:
            raise ValueError(
                "--validation-impostor-trials must be at least 1."
            )

    else:
        if args.enrol_recordings_per_speaker < 1:
            raise ValueError(
                "--enrol-recordings-per-speaker must be at least 1."
            )
        if args.min_verification_seconds <= 0:
            raise ValueError(
                "--min-verification-seconds must be greater than zero."
            )


def safe_id(path: Path, data_root: Path) -> str:
    relative = path.relative_to(data_root).with_suffix("")
    return re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        "--".join(relative.parts),
    )


def read_audio_info(
    path: Path,
    data_root: Path,
    sample_rate: int,
) -> dict:
    info = sf.info(str(path))
    if info.samplerate != sample_rate:
        raise ValueError(
            f"{path}: expected {sample_rate} Hz, got {info.samplerate} Hz"
        )
    if info.channels != 1:
        raise ValueError(
            f"{path}: expected mono audio, got {info.channels} channels"
        )
    if info.frames <= 0:
        raise ValueError(f"{path}: audio contains no frames")

    relative_path = path.relative_to(data_root)
    return {
        "path": path.resolve(),
        "relative_path": relative_path.as_posix(),
        "recording_id": safe_id(path, data_root),
        "spk_id": relative_path.parts[0],
        "frames": int(info.frames),
        "duration": float(info.frames / info.samplerate),
    }


def scan_dataset(
    data_root: Path,
    sample_rate: int,
) -> dict[str, list[dict]]:
    data_root = data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_root}")

    speakers: dict[str, list[dict]] = {}
    seen_ids: dict[str, Path] = {}

    for speaker_dir in sorted(data_root.iterdir()):
        if not speaker_dir.is_dir():
            continue

        recordings = []
        paths = sorted(
            path
            for path in speaker_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        )
        for path in paths:
            recording = read_audio_info(path, data_root, sample_rate)
            recording_id = recording["recording_id"]
            if recording_id in seen_ids:
                raise ValueError(
                    "Recording ID collision: "
                    f"{seen_ids[recording_id]} and {path}"
                )
            seen_ids[recording_id] = path
            recordings.append(recording)

        if recordings:
            speakers[speaker_dir.name] = recordings

    if not speakers:
        raise ValueError(f"No speaker audio found under: {data_root}")
    return speakers


def audit_row(recording: dict, split: str) -> dict:
    return {
        "recording_id": recording["recording_id"],
        "relative_path": recording["relative_path"],
        "spk_id": recording["spk_id"],
        "split": split,
        "duration": round(recording["duration"], 6),
        "frames": recording["frames"],
    }


def full_recording_row(recording: dict) -> dict:
    return {
        "ID": recording["recording_id"],
        "duration": round(recording["duration"], 6),
        "wav": str(recording["path"]),
        "start": 0,
        "stop": recording["frames"],
        "spk_id": recording["spk_id"],
        "_recording_id": recording["recording_id"],
    }


def make_chunk_rows(
    recordings: list[dict],
    speaker_id: str,
    sample_rate: int,
    chunk_seconds: float,
    min_chunk_seconds: float,
) -> list[dict]:
    chunk_frames = int(round(chunk_seconds * sample_rate))
    min_chunk_frames = int(round(min_chunk_seconds * sample_rate))
    rows: list[dict] = []

    for recording in recordings:
        start = 0
        segment_index = 0
        while start < recording["frames"]:
            remaining = recording["frames"] - start
            if remaining >= chunk_frames:
                stop = start + chunk_frames
            elif remaining >= min_chunk_frames:
                stop = recording["frames"]
            else:
                break

            rows.append(
                {
                    "ID": (
                        f"{recording['recording_id']}"
                        f"--seg{segment_index:04d}"
                    ),
                    "duration": round((stop - start) / sample_rate, 6),
                    "wav": str(recording["path"]),
                    "start": start,
                    "stop": stop,
                    "spk_id": speaker_id,
                    "_recording_id": recording["recording_id"],
                }
            )
            segment_index += 1
            start = stop
    return rows


def split_train_dev(
    speakers: dict[str, list[dict]],
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split complete recordings per speaker before making chunks."""
    train_rows: list[dict] = []
    dev_rows: list[dict] = []
    audit_rows: list[dict] = []

    for speaker_id in sorted(speakers):
        eligible = [
            recording
            for recording in speakers[speaker_id]
            if recording["duration"] >= args.min_chunk_seconds
        ]
        if len(eligible) < 2:
            raise ValueError(
                f"{speaker_id} needs at least two recordings of "
                f"{args.min_chunk_seconds} seconds or longer for train/dev."
            )

        rng.shuffle(eligible)
        num_dev = max(1, round(len(eligible) * args.dev_ratio))
        num_dev = min(num_dev, len(eligible) - 1)
        dev_recordings = eligible[:num_dev]
        train_recordings = eligible[num_dev:]

        speaker_train_rows = make_chunk_rows(
            train_recordings,
            speaker_id,
            args.sample_rate,
            args.chunk_seconds,
            args.min_chunk_seconds,
        )
        speaker_dev_rows = make_chunk_rows(
            dev_recordings,
            speaker_id,
            args.sample_rate,
            args.chunk_seconds,
            args.min_chunk_seconds,
        )
        if not speaker_train_rows or not speaker_dev_rows:
            raise ValueError(
                f"Could not create train and dev chunks for {speaker_id}."
            )

        train_rows.extend(speaker_train_rows)
        dev_rows.extend(speaker_dev_rows)
        audit_rows.extend(
            audit_row(recording, "train")
            for recording in train_recordings
        )
        audit_rows.extend(
            audit_row(recording, "dev")
            for recording in dev_recordings
        )

    return train_rows, dev_rows, audit_rows


def speaker_ranking_hash(seed: int, speaker_id: str) -> str:
    """Return a stable ranking key independent of filesystem ordering."""
    value = f"{seed}|{speaker_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def resolve_validation_speaker_count(
    total_speakers: int,
    exact_count: int | None,
    ratio: float | None,
) -> int:
    if total_speakers < 3:
        raise ValueError(
            "Validation EER needs at least three speakers: one train and "
            "two validation speakers."
        )

    if exact_count is not None:
        count = exact_count
    else:
        selected_ratio = (
            DEFAULT_VALIDATION_SPEAKER_RATIO if ratio is None else ratio
        )
        count = max(2, round(total_speakers * selected_ratio))

    if count >= total_speakers:
        raise ValueError(
            f"Requested {count} validation speakers from only "
            f"{total_speakers}; at least one training speaker is required."
        )
    return count


def split_train_validation_eer(
    speakers: dict[str, list[dict]],
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Split speakers, then build train chunks and validation utterances."""
    validation_count = resolve_validation_speaker_count(
        len(speakers),
        args.validation_speakers,
        args.validation_speaker_ratio,
    )

    ranked_speakers = sorted(
        speakers,
        key=lambda speaker_id: (
            speaker_ranking_hash(args.seed, speaker_id),
            speaker_id,
        ),
    )
    validation_candidates = [
        speaker_id
        for speaker_id in ranked_speakers
        if sum(
            recording["duration"] >= args.min_validation_seconds
            for recording in speakers[speaker_id]
        )
        >= 2
    ]
    if len(validation_candidates) < validation_count:
        raise ValueError(
            f"Only {len(validation_candidates)} speaker(s) have at least "
            "two validation-eligible recordings, but "
            f"{validation_count} validation speakers were requested."
        )

    validation_speakers = set(validation_candidates[:validation_count])
    train_speakers = set(speakers) - validation_speakers

    train_rows: list[dict] = []
    validation_rows: list[dict] = []
    audit_rows: list[dict] = []
    speaker_split_rows: list[dict] = []

    for speaker_id in sorted(speakers):
        split = (
            "validation"
            if speaker_id in validation_speakers
            else "train"
        )
        speaker_split_rows.append(
            {
                "spk_id": speaker_id,
                "split": split,
                "ranking_sha256": speaker_ranking_hash(
                    args.seed,
                    speaker_id,
                ),
            }
        )

        if speaker_id in validation_speakers:
            eligible = [
                recording
                for recording in speakers[speaker_id]
                if recording["duration"] >= args.min_validation_seconds
            ]
            validation_rows.extend(
                full_recording_row(recording) for recording in eligible
            )
            audit_rows.extend(
                audit_row(recording, "validation")
                for recording in eligible
            )
            continue

        eligible = [
            recording
            for recording in speakers[speaker_id]
            if recording["duration"] >= args.min_chunk_seconds
        ]
        speaker_rows = make_chunk_rows(
            eligible,
            speaker_id,
            args.sample_rate,
            args.chunk_seconds,
            args.min_chunk_seconds,
        )
        if not speaker_rows:
            raise ValueError(
                f"Training speaker {speaker_id} has no recording long "
                f"enough for a {args.min_chunk_seconds}-second chunk."
            )
        train_rows.extend(speaker_rows)
        audit_rows.extend(
            audit_row(recording, "train") for recording in eligible
        )

    return (
        train_rows,
        validation_rows,
        audit_rows,
        speaker_split_rows,
    )


def stable_digest(*parts: object) -> str:
    value = "\x1f".join(map(str, parts)).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def canonical_row_pair(left: dict, right: dict) -> tuple[dict, dict]:
    if left["ID"] == right["ID"]:
        raise ValueError("Validation self-pairs are forbidden.")
    if left["ID"] < right["ID"]:
        return left, right
    return right, left


def balanced_quotas(
    speakers: list[str],
    total: int,
    seed: int,
    purpose: str,
) -> dict[str, int]:
    base, remainder = divmod(total, len(speakers))
    ranked = sorted(
        speakers,
        key=lambda speaker: (
            stable_digest(seed, purpose, speaker),
            speaker,
        ),
    )
    return {
        speaker: base + int(index < remainder)
        for index, speaker in enumerate(ranked)
    }


def balanced_impostor_speaker_pairs(
    speakers: list[str],
    total: int,
    seed: int,
) -> list[tuple[str, str]]:
    remaining = balanced_quotas(
        speakers,
        total * 2,
        seed,
        "impostor-participation",
    )
    pairs: list[tuple[str, str]] = []
    for position in range(total):
        ranked = sorted(
            (
                speaker
                for speaker, count in remaining.items()
                if count > 0
            ),
            key=lambda speaker: (
                -remaining[speaker],
                stable_digest(
                    seed,
                    "impostor-speaker",
                    position,
                    speaker,
                ),
                speaker,
            ),
        )
        if len(ranked) < 2:
            raise ValueError(
                "Could not balance impostor participation across "
                "validation speakers."
            )
        left_speaker, right_speaker = ranked[:2]
        remaining[left_speaker] -= 1
        remaining[right_speaker] -= 1
        pairs.append((left_speaker, right_speaker))

    if any(remaining.values()):
        raise RuntimeError("Impostor speaker quotas did not reconcile.")
    return pairs


def create_validation_trials(
    validation_rows: list[dict],
    genuine_count: int,
    impostor_count: int,
    trial_seed: int,
) -> list[dict]:
    """Create speaker-balanced trials using stable hash ranking."""
    rows_by_speaker: dict[str, list[dict]] = {}
    for row in validation_rows:
        rows_by_speaker.setdefault(row["spk_id"], []).append(row)
    speakers = sorted(rows_by_speaker)
    if len(speakers) < 2:
        raise ValueError(
            "At least two validation speakers are required for EER."
        )
    for rows in rows_by_speaker.values():
        rows.sort(key=lambda row: row["ID"])

    selected: list[tuple[int, dict, dict]] = []
    seen_pairs: set[tuple[str, str]] = set()

    genuine_quotas = balanced_quotas(
        speakers,
        genuine_count,
        trial_seed,
        "genuine-quota",
    )
    for speaker in speakers:
        rows = rows_by_speaker[speaker]
        candidates = (
            (
                stable_digest(
                    trial_seed,
                    "genuine",
                    speaker,
                    left["ID"],
                    right["ID"],
                ),
                left,
                right,
            )
            for offset, left in enumerate(rows)
            for right in rows[offset + 1 :]
        )
        quota = genuine_quotas[speaker]
        ranked = heapq.nsmallest(quota, candidates, key=lambda item: item[0])
        if len(ranked) != quota:
            capacity = len(rows) * (len(rows) - 1) // 2
            raise ValueError(
                f"Validation speaker {speaker} can supply only {capacity} "
                f"genuine pair(s), but its balanced quota is {quota}. "
                "Reduce --validation-genuine-trials or add recordings."
            )
        for _, left, right in ranked:
            left, right = canonical_row_pair(left, right)
            pair_id = (left["ID"], right["ID"])
            if pair_id in seen_pairs:
                raise RuntimeError("Duplicate validation genuine pair.")
            seen_pairs.add(pair_id)
            selected.append((1, left, right))

    utterance_use: Counter[str] = Counter()
    speaker_pairs = balanced_impostor_speaker_pairs(
        speakers,
        impostor_count,
        trial_seed,
    )
    for position, (left_speaker, right_speaker) in enumerate(speaker_pairs):
        left_rows = sorted(
            rows_by_speaker[left_speaker],
            key=lambda row: (
                utterance_use[row["ID"]],
                stable_digest(
                    trial_seed,
                    "impostor",
                    position,
                    left_speaker,
                    row["ID"],
                ),
                row["ID"],
            ),
        )
        right_rows = sorted(
            rows_by_speaker[right_speaker],
            key=lambda row: (
                utterance_use[row["ID"]],
                stable_digest(
                    trial_seed,
                    "impostor",
                    position,
                    right_speaker,
                    row["ID"],
                ),
                row["ID"],
            ),
        )

        chosen: tuple[dict, dict] | None = None
        for left in left_rows:
            for right in right_rows:
                canonical_left, canonical_right = canonical_row_pair(
                    left,
                    right,
                )
                if (
                    canonical_left["ID"],
                    canonical_right["ID"],
                ) not in seen_pairs:
                    chosen = (canonical_left, canonical_right)
                    break
            if chosen is not None:
                break
        if chosen is None:
            raise ValueError(
                "Validation impostor-pair capacity was exhausted. Reduce "
                "--validation-impostor-trials or add recordings."
            )

        left, right = chosen
        pair_id = (left["ID"], right["ID"])
        seen_pairs.add(pair_id)
        utterance_use[left["ID"]] += 1
        utterance_use[right["ID"]] += 1
        selected.append((0, left, right))

    return [
        {
            "trial_id": f"validation-{index:06d}",
            "target": target,
            "left_id": left["ID"],
            "right_id": right["ID"],
            "left_spk_id": left["spk_id"],
            "right_spk_id": right["spk_id"],
        }
        for index, (target, left, right) in enumerate(selected)
    ]


def split_enrol_test(
    speakers: dict[str, list[dict]],
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split complete external-test recordings into enrol and test."""
    if len(speakers) < 2:
        raise ValueError(
            "External verification requires at least two speakers."
        )

    enrol_rows: list[dict] = []
    test_rows: list[dict] = []
    audit_rows: list[dict] = []
    required = args.enrol_recordings_per_speaker + 1

    for speaker_id in sorted(speakers):
        eligible = [
            recording
            for recording in speakers[speaker_id]
            if recording["duration"] >= args.min_verification_seconds
        ]
        if len(eligible) < required:
            raise ValueError(
                f"{speaker_id} has {len(eligible)} eligible recording(s), "
                f"but needs at least {required}: "
                f"{args.enrol_recordings_per_speaker} enrol + 1 test."
            )

        rng.shuffle(eligible)
        enrol_recordings = eligible[
            : args.enrol_recordings_per_speaker
        ]
        test_recordings = eligible[
            args.enrol_recordings_per_speaker :
        ]

        enrol_rows.extend(
            full_recording_row(recording)
            for recording in enrol_recordings
        )
        test_rows.extend(
            full_recording_row(recording)
            for recording in test_recordings
        )
        audit_rows.extend(
            audit_row(recording, "enrol")
            for recording in enrol_recordings
        )
        audit_rows.extend(
            audit_row(recording, "test")
            for recording in test_recordings
        )

    return enrol_rows, test_rows, audit_rows


def create_trials(
    enrol_rows: list[dict],
    test_rows: list[dict],
    rng: random.Random,
    all_impostor_trials: bool,
) -> list[tuple[int, str, str]]:
    """Create genuine and impostor recording-level trials."""
    genuine: list[tuple[int, str, str]] = []
    impostor: list[tuple[int, str, str]] = []

    for enrol in enrol_rows:
        for test in test_rows:
            label = int(enrol["spk_id"] == test["spk_id"])
            trial = (label, enrol["ID"], test["ID"])
            if label == 1:
                genuine.append(trial)
            else:
                impostor.append(trial)

    if not genuine or not impostor:
        raise ValueError(
            "Verification trials need genuine and impostor pairs."
        )

    if not all_impostor_trials:
        rng.shuffle(impostor)
        impostor = impostor[: len(genuine)]

    trials = genuine + impostor
    rng.shuffle(trials)
    return trials


def read_speaker_ids(csv_path: Path) -> set[str]:
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Training manifest not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or "spk_id" not in reader.fieldnames:
            raise ValueError(f"{csv_path} has no spk_id column")
        speaker_ids = {
            row["spk_id"].strip()
            for row in reader
            if row["spk_id"].strip()
        }

    if not speaker_ids:
        raise ValueError(f"No speaker IDs found in: {csv_path}")
    return speaker_ids


def check_external_speakers(
    test_speakers: set[str],
    reference_csvs: list[Path | None],
) -> None:
    """Optionally verify train/validation/test speaker-ID separation."""
    for csv_path in reference_csvs:
        if csv_path is None:
            continue
        overlap = sorted(test_speakers & read_speaker_ids(csv_path))
        if overlap:
            raise ValueError(
                "External-test speaker IDs overlap with "
                f"{csv_path}: {overlap[:10]}"
            )


def validate_manifest_rows(
    split_rows: dict[str, list[dict]],
    sample_rate: int,
) -> None:
    all_ids: list[str] = []

    for split_name, rows in split_rows.items():
        if not rows:
            raise ValueError(f"{split_name}.csv would be empty")

        split_ids = [row["ID"] for row in rows]
        if len(split_ids) != len(set(split_ids)):
            raise ValueError(f"Duplicate IDs inside {split_name}.csv")
        all_ids.extend(split_ids)

        for row in rows:
            start = int(row["start"])
            stop = int(row["stop"])
            duration = float(row["duration"])
            if start < 0 or stop <= start or duration <= 0:
                raise ValueError(
                    f"Invalid manifest row in {split_name}: {row['ID']}"
                )

            expected_duration = (stop - start) / sample_rate
            if abs(duration - expected_duration) > 1e-5:
                raise ValueError(
                    f"Duration disagrees with start/stop in "
                    f"{split_name}: {row['ID']}"
                )

    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Duplicate IDs found across manifest files")


def validate_train_split(
    train_rows: list[dict],
    dev_rows: list[dict],
    sample_rate: int,
) -> None:
    validate_manifest_rows(
        {"train": train_rows, "dev": dev_rows},
        sample_rate,
    )

    train_speakers = {row["spk_id"] for row in train_rows}
    dev_speakers = {row["spk_id"] for row in dev_rows}
    if train_speakers != dev_speakers:
        raise ValueError(
            "train.csv and dev.csv must contain the same speakers"
        )

    train_recordings = {row["_recording_id"] for row in train_rows}
    dev_recordings = {row["_recording_id"] for row in dev_rows}
    overlap = sorted(train_recordings & dev_recordings)
    if overlap:
        raise ValueError(
            f"Recording leakage between train and dev: {overlap[:10]}"
        )


def validate_train_eer_split(
    train_rows: list[dict],
    validation_rows: list[dict],
    trials: list[dict],
    sample_rate: int,
) -> None:
    validate_manifest_rows(
        {"train": train_rows, "validation": validation_rows},
        sample_rate,
    )

    train_speakers = {row["spk_id"] for row in train_rows}
    validation_speakers = {
        row["spk_id"] for row in validation_rows
    }
    speaker_overlap = sorted(train_speakers & validation_speakers)
    if speaker_overlap:
        raise ValueError(
            "Speaker leakage between train and validation: "
            f"{speaker_overlap[:10]}"
        )
    if len(validation_speakers) < 2:
        raise ValueError(
            "Validation EER needs at least two validation speakers."
        )

    rows_by_id = {row["ID"]: row for row in validation_rows}
    trial_ids: set[str] = set()
    pair_ids: set[tuple[str, str]] = set()
    targets: set[int] = set()

    for trial in trials:
        trial_id = trial["trial_id"]
        if trial_id in trial_ids:
            raise ValueError(f"Duplicate validation trial ID: {trial_id}")
        trial_ids.add(trial_id)

        left_id = trial["left_id"]
        right_id = trial["right_id"]
        if left_id not in rows_by_id or right_id not in rows_by_id:
            raise ValueError(
                "Validation trial references an unknown utterance: "
                f"{left_id}, {right_id}"
            )
        pair_id = tuple(sorted((left_id, right_id)))
        if pair_id in pair_ids:
            raise ValueError(
                f"Duplicate validation pair: {pair_id[0]}, {pair_id[1]}"
            )
        pair_ids.add(pair_id)

        target = int(trial["target"])
        targets.add(target)
        expected = int(
            rows_by_id[left_id]["spk_id"]
            == rows_by_id[right_id]["spk_id"]
        )
        if target != expected:
            raise ValueError(
                f"Incorrect validation target: {left_id}, {right_id}"
            )

    if targets != {0, 1}:
        raise ValueError(
            "Validation trials must contain genuine and impostor pairs."
        )


def validate_test_split(
    enrol_rows: list[dict],
    test_rows: list[dict],
    trials: list[tuple[int, str, str]],
    sample_rate: int,
) -> None:
    validate_manifest_rows(
        {"enrol": enrol_rows, "test": test_rows},
        sample_rate,
    )

    enrol_speakers = {row["spk_id"] for row in enrol_rows}
    test_speakers = {row["spk_id"] for row in test_rows}
    if enrol_speakers != test_speakers:
        raise ValueError(
            "enrol.csv and test.csv must contain the same speakers"
        )

    enrol_recordings = {row["_recording_id"] for row in enrol_rows}
    test_recordings = {row["_recording_id"] for row in test_rows}
    overlap = sorted(enrol_recordings & test_recordings)
    if overlap:
        raise ValueError(
            f"Recording leakage between enrol and test: {overlap[:10]}"
        )

    enrol_by_id = {row["ID"]: row for row in enrol_rows}
    test_by_id = {row["ID"]: row for row in test_rows}
    for label, enrol_id, test_id in trials:
        if enrol_id not in enrol_by_id or test_id not in test_by_id:
            raise ValueError(
                f"Trial references unknown ID: {enrol_id}, {test_id}"
            )
        expected = int(
            enrol_by_id[enrol_id]["spk_id"]
            == test_by_id[test_id]["spk_id"]
        )
        if label != expected:
            raise ValueError(
                f"Incorrect trial label: {enrol_id}, {test_id}"
            )


def prepare_output_files(
    output_dir: Path,
    filenames: list[str],
    overwrite: bool,
) -> Path:
    """Create output_dir without deleting it or unrelated files."""
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = [
        output_dir / name
        for name in filenames
        if (output_dir / name).exists()
    ]
    if existing and not overwrite:
        formatted = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Manifest files already exist. Use --overwrite to replace "
            f"only these files:\n{formatted}"
        )
    return output_dir


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=MANIFEST_COLUMNS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_audit(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_dict_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_trials(
    path: Path,
    trials: list[tuple[int, str, str]],
) -> None:
    with path.open("w", encoding="utf-8") as file:
        for label, enrol_id, test_id in trials:
            file.write(f"{label} {enrol_id} {test_id}\n")


def run_train(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    speakers = scan_dataset(args.data_root, args.sample_rate)
    train_rows, dev_rows, audit_rows = split_train_dev(
        speakers,
        args,
        rng,
    )
    validate_train_split(train_rows, dev_rows, args.sample_rate)

    output_dir = prepare_output_files(
        args.output_dir,
        ["train.csv", "dev.csv", "recording_split.csv"],
        args.overwrite,
    )
    write_csv(output_dir / "train.csv", train_rows)
    write_csv(output_dir / "dev.csv", dev_rows)
    write_audit(output_dir / "recording_split.csv", audit_rows)

    print("Train/validation manifests created")
    print(f"  Speakers:     {len(speakers)}")
    print(f"  Train chunks: {len(train_rows)}")
    print(f"  Dev chunks:   {len(dev_rows)}")
    print(f"  Output:        {output_dir}")


def run_train_eer(args: argparse.Namespace) -> None:
    speakers = scan_dataset(args.data_root, args.sample_rate)
    (
        train_rows,
        validation_rows,
        audit_rows,
        speaker_split_rows,
    ) = split_train_validation_eer(speakers, args)
    trials = create_validation_trials(
        validation_rows,
        genuine_count=args.validation_genuine_trials,
        impostor_count=args.validation_impostor_trials,
        trial_seed=args.trial_seed,
    )
    validate_train_eer_split(
        train_rows,
        validation_rows,
        trials,
        args.sample_rate,
    )

    output_dir = prepare_output_files(
        args.output_dir,
        [
            "train.csv",
            "validation.csv",
            "validation_trials.csv",
            "recording_split.csv",
            "speaker_split.csv",
        ],
        args.overwrite,
    )
    write_csv(output_dir / "train.csv", train_rows)
    write_csv(output_dir / "validation.csv", validation_rows)
    write_dict_csv(
        output_dir / "validation_trials.csv",
        trials,
        VALIDATION_TRIAL_COLUMNS,
    )
    write_audit(output_dir / "recording_split.csv", audit_rows)
    write_dict_csv(
        output_dir / "speaker_split.csv",
        speaker_split_rows,
        SPEAKER_SPLIT_COLUMNS,
    )

    train_speaker_count = len({row["spk_id"] for row in train_rows})
    validation_speaker_count = len(
        {row["spk_id"] for row in validation_rows}
    )
    genuine = sum(int(row["target"]) == 1 for row in trials)
    impostor = sum(int(row["target"]) == 0 for row in trials)
    print("Speaker-disjoint validation-EER manifests created")
    print(f"  Train speakers:          {train_speaker_count}")
    print(f"  Validation speakers:     {validation_speaker_count}")
    print(f"  Train chunks:            {len(train_rows)}")
    print(f"  Validation recordings:  {len(validation_rows)}")
    print(f"  Validation genuine:      {genuine}")
    print(f"  Validation impostor:     {impostor}")
    print(f"  Output:                  {output_dir}")


def run_test(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    speakers = scan_dataset(args.data_root, args.sample_rate)
    check_external_speakers(
        set(speakers),
        [args.train_csv, args.validation_csv],
    )

    enrol_rows, test_rows, audit_rows = split_enrol_test(
        speakers,
        args,
        rng,
    )
    trials = create_trials(
        enrol_rows,
        test_rows,
        rng,
        all_impostor_trials=args.all_impostor_trials,
    )
    validate_test_split(
        enrol_rows,
        test_rows,
        trials,
        args.sample_rate,
    )

    output_dir = prepare_output_files(
        args.output_dir,
        [
            "enrol.csv",
            "test.csv",
            "verification_trials.txt",
            "recording_split.csv",
        ],
        args.overwrite,
    )
    write_csv(output_dir / "enrol.csv", enrol_rows)
    write_csv(output_dir / "test.csv", test_rows)
    write_trials(output_dir / "verification_trials.txt", trials)
    write_audit(output_dir / "recording_split.csv", audit_rows)

    genuine = sum(label == 1 for label, _, _ in trials)
    impostor = sum(label == 0 for label, _, _ in trials)
    print("External-test manifests created")
    print(f"  Speakers:         {len(speakers)}")
    print(f"  Enrol recordings: {len(enrol_rows)}")
    print(f"  Test recordings:  {len(test_rows)}")
    print(f"  Genuine trials:   {genuine}")
    print(f"  Impostor trials:  {impostor}")
    print(f"  Output:            {output_dir}")


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        if args.command == "train":
            run_train(args)
        elif args.command == "train-eer":
            run_train_eer(args)
        else:
            run_test(args)
        return 0
    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
