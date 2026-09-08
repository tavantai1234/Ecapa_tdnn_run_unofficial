#!/usr/bin/env python3
"""Create SpeechBrain manifests from separate train/validation and test data.

Examples
--------
Train/validation manifests:

    python prepare_data.py train \
        --data-root processed/train_val \
        --output-dir manifests/train_val

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
import random
import re
import sys
from pathlib import Path

import soundfile as sf


DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_SEED = 42
DEFAULT_DEV_RATIO = 0.10
DEFAULT_CHUNK_SECONDS = 3.0
DEFAULT_MIN_CHUNK_SECONDS = 1.5
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
        description="Prepare train/dev or external verification manifests."
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
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be greater than zero.")

    if args.command == "train":
        if not 0.0 < args.dev_ratio < 1.0:
            raise ValueError("--dev-ratio must be between 0 and 1.")
        if args.chunk_seconds <= 0 or args.min_chunk_seconds <= 0:
            raise ValueError("Chunk durations must be greater than zero.")
        if args.min_chunk_seconds > args.chunk_seconds:
            raise ValueError(
                "--min-chunk-seconds cannot exceed --chunk-seconds."
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
    train_csv: Path | None,
) -> None:
    """Optionally verify train/test speaker-ID separation."""
    if train_csv is None:
        return

    overlap = sorted(test_speakers & read_speaker_ids(train_csv))
    if overlap:
        raise ValueError(
            "Training and external-test speaker IDs overlap: "
            f"{overlap[:10]}"
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


def run_test(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    speakers = scan_dataset(args.data_root, args.sample_rate)
    check_external_speakers(set(speakers), args.train_csv)

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
