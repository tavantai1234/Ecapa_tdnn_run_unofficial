#!/usr/bin/env python3
"""Prepare SpeechBrain manifests for the ECAPA-TDNN baseline E0.

Input
-----
Audio processed by ``preprocess_e0.py``:

    processed/E0/
        speaker_01/*.wav
        speaker_02/*.wav
        ...

Every input recording must already be:
- mono;
- 16 kHz;
- otherwise kept at its original duration.

Output
------
    manifests/E0/
        train.csv
        dev.csv
        enrol.csv
        test.csv
        verification_trials.txt
        recording_split.csv

Important design
----------------
1. Speakers are divided into training speakers and unseen verification speakers.
2. Original recordings are assigned to train/dev or enrol/test BEFORE chunking.
3. Train/dev recordings are represented by chunks using CSV ``start``/``stop``.
4. No chunk WAV files are created.
5. Residual train/dev chunks from 1.5 to under 3 seconds are retained. SpeechBrain
   pads them when creating a batch.
6. Enrol/test rows represent complete, mutually disjoint recordings. This keeps
   verification trials at the utterance/recording level.
"""

from __future__ import annotations

import csv
import random
import re
import shutil
from pathlib import Path

import soundfile as sfgit 


# ======================== CONFIG ========================
PROJECT_ROOT = Path(__file__).resolve().parent

# This is the output directory created by preprocess_e0.py.
DATASET_ROOT = PROJECT_ROOT / "processed" / "E0"

# Keep E0 manifests separate from later E1/E2 experiments.
OUTPUT_DIR = PROJECT_ROOT / "manifests" / "E0"

SAMPLE_RATE = 16_000

# Train/dev chunk configuration.
CHUNK_SECONDS = 3.0
MIN_CHUNK_SECONDS = 1.5

# Recording-level split ratio for every training speaker.
DEV_RATIO = 0.10

NUM_VERIFICATION_SPEAKERS = 2
ENROL_RECORDINGS_PER_SPEAKER = 5

# Verification recordings shorter than this are ignored.
MIN_VERIFICATION_SECONDS = 1.5

# None: choose verification speakers reproducibly using SEED.
# Example: ["speaker_06", "speaker_07"]
VERIFICATION_SPEAKERS = None

SEED = 42
AUDIO_EXTENSIONS = {".wav", ".flac"}


MANIFEST_COLUMNS = [
    "ID",
    "duration",
    "wav",
    "start",
    "stop",
    "spk_id",
]


# ======================== FILESYSTEM ========================

def reset_output_dir() -> None:
    """Safely recreate only ``manifests/E0``."""
    output_dir = OUTPUT_DIR.expanduser().resolve()

    if output_dir.name != "E0" or output_dir.parent.name != "manifests":
        raise ValueError(
            "Safety check failed: OUTPUT_DIR must point to "
            "'.../manifests/E0'."
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=False)


# ======================== AUDIO SCAN ========================

def safe_id(path: Path) -> str:
    """Create a stable ID from the path relative to DATASET_ROOT."""
    relative = path.relative_to(DATASET_ROOT).with_suffix("")
    raw = "--".join(relative.parts)
    return re.sub(r"[^A-Za-z0-9_-]+", "_", raw)


def read_audio_info(path: Path) -> dict:
    """Read and validate one processed recording."""
    info = sf.info(str(path))

    if info.samplerate != SAMPLE_RATE:
        raise ValueError(
            f"{path}: expected {SAMPLE_RATE} Hz, got {info.samplerate} Hz"
        )

    if info.channels != 1:
        raise ValueError(
            f"{path}: expected mono audio, got {info.channels} channels"
        )

    if info.frames <= 0:
        raise ValueError(f"{path}: audio contains no frames")

    relative_path = path.relative_to(DATASET_ROOT)
    speaker_id = relative_path.parts[0]

    return {
        "path": path.resolve(),
        "relative_path": relative_path.as_posix(),
        "recording_id": safe_id(path),
        "spk_id": speaker_id,
        "frames": int(info.frames),
        "duration": float(info.frames / info.samplerate),
    }


def scan_dataset() -> dict[str, list[dict]]:
    """Scan one top-level directory per speaker."""
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATASET_ROOT}\n"
            "Run preprocess_e0.py first."
        )

    speakers: dict[str, list[dict]] = {}

    for speaker_dir in sorted(DATASET_ROOT.iterdir()):
        if not speaker_dir.is_dir():
            continue

        paths = sorted(
            path
            for path in speaker_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in AUDIO_EXTENSIONS
        )

        if paths:
            speakers[speaker_dir.name] = [
                read_audio_info(path) for path in paths
            ]

    if len(speakers) < NUM_VERIFICATION_SPEAKERS + 1:
        raise ValueError(
            "At least one training speaker and "
            f"{NUM_VERIFICATION_SPEAKERS} verification speakers are required."
        )

    return speakers


# ======================== SPEAKER SPLIT ========================

def training_eligible(recording: dict) -> bool:
    """Whether a recording can produce at least one retained chunk."""
    return recording["duration"] >= MIN_CHUNK_SECONDS


def verification_eligible(recording: dict) -> bool:
    """Whether a recording is long enough for enrol/test."""
    return recording["duration"] >= MIN_VERIFICATION_SECONDS


def choose_speakers(
    speakers: dict[str, list[dict]],
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    """Create disjoint training and unseen verification speaker groups."""
    required_verification_recordings = ENROL_RECORDINGS_PER_SPEAKER + 1

    eligible_verification_speakers = sorted(
        speaker_id
        for speaker_id, recordings in speakers.items()
        if sum(
            verification_eligible(recording)
            for recording in recordings
        )
        >= required_verification_recordings
    )

    if VERIFICATION_SPEAKERS is None:
        if (
            len(eligible_verification_speakers)
            < NUM_VERIFICATION_SPEAKERS
        ):
            raise ValueError(
                "Not enough speakers have sufficient recordings for "
                "enrol/test."
            )

        verification_speakers = sorted(
            rng.sample(
                eligible_verification_speakers,
                NUM_VERIFICATION_SPEAKERS,
            )
        )
    else:
        verification_speakers = sorted(VERIFICATION_SPEAKERS)

        if (
            len(verification_speakers)
            != NUM_VERIFICATION_SPEAKERS
        ):
            raise ValueError(
                "VERIFICATION_SPEAKERS length must equal "
                "NUM_VERIFICATION_SPEAKERS."
            )

        unknown = set(verification_speakers) - set(speakers)
        if unknown:
            raise ValueError(
                f"Unknown verification speakers: {sorted(unknown)}"
            )

        insufficient = [
            speaker_id
            for speaker_id in verification_speakers
            if sum(
                verification_eligible(recording)
                for recording in speakers[speaker_id]
            )
            < required_verification_recordings
        ]
        if insufficient:
            raise ValueError(
                "Verification speakers do not have enough eligible "
                f"recordings: {insufficient}"
            )

    training_speakers = sorted(
        set(speakers) - set(verification_speakers)
    )

    return training_speakers, verification_speakers


# ======================== CHUNK CREATION ========================

def make_chunk_rows(
    recordings: list[dict],
    speaker_id: str,
) -> list[dict]:
    """Create train/dev CSV rows without writing physical chunk files."""
    chunk_frames = int(round(CHUNK_SECONDS * SAMPLE_RATE))
    min_chunk_frames = int(round(MIN_CHUNK_SECONDS * SAMPLE_RATE))

    rows: list[dict] = []

    for recording in recordings:
        total_frames = recording["frames"]
        segment_index = 0
        start = 0

        while start < total_frames:
            remaining = total_frames - start

            if remaining >= chunk_frames:
                stop = start + chunk_frames
            elif remaining >= min_chunk_frames:
                # Retain the final short chunk. PaddedBatch will pad it later.
                stop = total_frames
            else:
                # Discard a residual shorter than MIN_CHUNK_SECONDS.
                break

            duration = (stop - start) / SAMPLE_RATE

            rows.append(
                {
                    "ID": (
                        f"{recording['recording_id']}"
                        f"--seg{segment_index:04d}"
                    ),
                    "duration": round(duration, 6),
                    "wav": str(recording["path"]),
                    "start": start,
                    "stop": stop,
                    "spk_id": speaker_id,
                    # Internal field for leakage validation.
                    "_recording_id": recording["recording_id"],
                }
            )

            segment_index += 1
            start = stop

    return rows


# ======================== RECORDING SPLITS ========================

def split_train_dev(
    speakers: dict[str, list[dict]],
    training_speakers: list[str],
    rng: random.Random,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split complete recordings first, then create train/dev chunks.

    Returns:
        train chunk rows,
        dev chunk rows,
        recording-level audit rows.
    """
    train_rows: list[dict] = []
    dev_rows: list[dict] = []
    audit_rows: list[dict] = []

    for speaker_id in training_speakers:
        eligible = [
            recording
            for recording in speakers[speaker_id]
            if training_eligible(recording)
        ]

        if len(eligible) < 2:
            raise ValueError(
                f"{speaker_id} needs at least two recordings of "
                f"{MIN_CHUNK_SECONDS} seconds or longer for train/dev."
            )

        rng.shuffle(eligible)

        num_dev = max(1, round(len(eligible) * DEV_RATIO))
        num_dev = min(num_dev, len(eligible) - 1)

        dev_recordings = eligible[:num_dev]
        train_recordings = eligible[num_dev:]

        speaker_train_rows = make_chunk_rows(
            train_recordings,
            speaker_id,
        )
        speaker_dev_rows = make_chunk_rows(
            dev_recordings,
            speaker_id,
        )

        if not speaker_train_rows or not speaker_dev_rows:
            raise ValueError(
                f"Could not create both train and dev chunks for "
                f"{speaker_id}."
            )

        train_rows.extend(speaker_train_rows)
        dev_rows.extend(speaker_dev_rows)

        audit_rows.extend(
            make_recording_audit_row(recording, "train")
            for recording in train_recordings
        )
        audit_rows.extend(
            make_recording_audit_row(recording, "dev")
            for recording in dev_recordings
        )

    return train_rows, dev_rows, audit_rows


def make_full_recording_row(recording: dict) -> dict:
    """Create one enrol/test row for a complete recording."""
    return {
        "ID": recording["recording_id"],
        "duration": round(recording["duration"], 6),
        "wav": str(recording["path"]),
        "start": 0,
        "stop": recording["frames"],
        "spk_id": recording["spk_id"],
        "_recording_id": recording["recording_id"],
    }


def split_enrol_test(
    speakers: dict[str, list[dict]],
    verification_speakers: list[str],
    rng: random.Random,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split complete recordings into mutually disjoint enrol/test sets."""
    eligible_by_speaker = {
        speaker_id: [
            recording
            for recording in speakers[speaker_id]
            if verification_eligible(recording)
        ]
        for speaker_id in verification_speakers
    }

    balanced_count = min(
        len(recordings)
        for recordings in eligible_by_speaker.values()
    )

    if balanced_count <= ENROL_RECORDINGS_PER_SPEAKER:
        raise ValueError(
            "Verification speakers need more eligible recordings than "
            "ENROL_RECORDINGS_PER_SPEAKER."
        )

    enrol_rows: list[dict] = []
    test_rows: list[dict] = []
    audit_rows: list[dict] = []

    for speaker_id in verification_speakers:
        selected = list(eligible_by_speaker[speaker_id])
        rng.shuffle(selected)
        selected = selected[:balanced_count]

        enrol_recordings = selected[:ENROL_RECORDINGS_PER_SPEAKER]
        test_recordings = selected[ENROL_RECORDINGS_PER_SPEAKER:]

        enrol_rows.extend(
            make_full_recording_row(recording)
            for recording in enrol_recordings
        )
        test_rows.extend(
            make_full_recording_row(recording)
            for recording in test_recordings
        )

        audit_rows.extend(
            make_recording_audit_row(recording, "enrol")
            for recording in enrol_recordings
        )
        audit_rows.extend(
            make_recording_audit_row(recording, "test")
            for recording in test_recordings
        )

    return enrol_rows, test_rows, audit_rows


def make_recording_audit_row(
    recording: dict,
    split: str,
) -> dict:
    """Create one row showing the recording-level assignment."""
    return {
        "recording_id": recording["recording_id"],
        "relative_path": recording["relative_path"],
        "spk_id": recording["spk_id"],
        "split": split,
        "duration": round(recording["duration"], 6),
        "frames": recording["frames"],
    }


# ======================== VERIFICATION TRIALS ========================

def create_trials(
    enrol_rows: list[dict],
    test_rows: list[dict],
    rng: random.Random,
) -> list[tuple[int, str, str]]:
    """Create balanced genuine and impostor recording-level trials."""
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
            "Verification trials need both genuine and impostor pairs."
        )

    rng.shuffle(impostor)
    impostor = impostor[:len(genuine)]

    trials = genuine + impostor
    rng.shuffle(trials)

    return trials


# ======================== VALIDATION ========================

def validate_manifest_row(row: dict, split_name: str) -> None:
    start = int(row["start"])
    stop = int(row["stop"])
    duration = float(row["duration"])

    if start < 0:
        raise ValueError(
            f"{split_name}: negative start in {row['ID']}"
        )
    if stop <= start:
        raise ValueError(
            f"{split_name}: invalid start/stop in {row['ID']}"
        )
    if duration <= 0:
        raise ValueError(
            f"{split_name}: non-positive duration in {row['ID']}"
        )

    expected_duration = (stop - start) / SAMPLE_RATE
    if abs(duration - expected_duration) > 1e-5:
        raise ValueError(
            f"{split_name}: duration disagrees with start/stop "
            f"in {row['ID']}"
        )


def validate_split(
    train_rows: list[dict],
    dev_rows: list[dict],
    enrol_rows: list[dict],
    test_rows: list[dict],
) -> None:
    """Check speaker separation, recording leakage, and duplicate IDs."""
    split_rows = {
        "train": train_rows,
        "dev": dev_rows,
        "enrol": enrol_rows,
        "test": test_rows,
    }

    for split_name, rows in split_rows.items():
        if not rows:
            raise ValueError(f"{split_name}.csv would be empty")

        ids = [row["ID"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(
                f"Duplicate IDs found inside {split_name}.csv"
            )

        for row in rows:
            validate_manifest_row(row, split_name)

    all_ids = [
        row["ID"]
        for rows in split_rows.values()
        for row in rows
    ]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Duplicate IDs found across manifest files")

    train_speakers = {row["spk_id"] for row in train_rows}
    dev_speakers = {row["spk_id"] for row in dev_rows}
    verification_speakers = {
        row["spk_id"]
        for row in enrol_rows + test_rows
    }

    if train_speakers != dev_speakers:
        raise ValueError(
            "train.csv and dev.csv must contain the same speakers"
        )

    if not train_speakers.isdisjoint(verification_speakers):
        raise ValueError(
            "Training and verification speaker groups overlap"
        )

    train_recordings = {
        row["_recording_id"] for row in train_rows
    }
    dev_recordings = {
        row["_recording_id"] for row in dev_rows
    }
    enrol_recordings = {
        row["_recording_id"] for row in enrol_rows
    }
    test_recordings = {
        row["_recording_id"] for row in test_rows
    }

    if not train_recordings.isdisjoint(dev_recordings):
        overlap = sorted(train_recordings & dev_recordings)
        raise ValueError(
            f"Recording leakage between train and dev: {overlap[:5]}"
        )

    if not enrol_recordings.isdisjoint(test_recordings):
        overlap = sorted(enrol_recordings & test_recordings)
        raise ValueError(
            f"Recording leakage between enrol and test: {overlap[:5]}"
        )


# ======================== WRITING ========================

def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a SpeechBrain manifest without internal validation fields."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=MANIFEST_COLUMNS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_recording_split(
    path: Path,
    rows: list[dict],
) -> None:
    """Write a reproducibility/audit table at original-recording level."""
    columns = [
        "recording_id",
        "relative_path",
        "spk_id",
        "split",
        "duration",
        "frames",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_trials(
    path: Path,
    trials: list[tuple[int, str, str]],
) -> None:
    """Write ``label enrol_id test_id``."""
    with path.open("w", encoding="utf-8") as file:
        for label, enrol_id, test_id in trials:
            file.write(f"{label} {enrol_id} {test_id}\n")


# ======================== SUMMARY ========================

def print_summary(
    speakers: dict[str, list[dict]],
    training_speakers: list[str],
    verification_speakers: list[str],
    train_rows: list[dict],
    dev_rows: list[dict],
    enrol_rows: list[dict],
    test_rows: list[dict],
    recording_split_rows: list[dict],
    trials: list[tuple[int, str, str]],
) -> None:
    print("\nRecordings discovered per speaker:")
    for speaker_id in sorted(speakers):
        role = (
            "verification"
            if speaker_id in verification_speakers
            else "training"
        )
        print(
            f"  {speaker_id:<20} "
            f"{len(speakers[speaker_id]):>4} recordings [{role}]"
        )

    split_recording_counts = {
        split: sum(
            row["split"] == split
            for row in recording_split_rows
        )
        for split in ("train", "dev", "enrol", "test")
    }

    genuine_count = sum(label == 1 for label, _, _ in trials)
    impostor_count = sum(label == 0 for label, _, _ in trials)

    print("\nSpeaker split:")
    print("  Training:    ", ", ".join(training_speakers))
    print("  Verification:", ", ".join(verification_speakers))

    print("\nRecording assignments:")
    for split in ("train", "dev", "enrol", "test"):
        print(
            f"  {split:<6}: "
            f"{split_recording_counts[split]} recordings"
        )

    print("\nGenerated manifests:")
    print(f"  train.csv:               {len(train_rows)} chunks")
    print(f"  dev.csv:                 {len(dev_rows)} chunks")
    print(f"  enrol.csv:               {len(enrol_rows)} recordings")
    print(f"  test.csv:                {len(test_rows)} recordings")
    print(f"  verification_trials.txt: {len(trials)} trials")
    print(f"    genuine:               {genuine_count}")
    print(f"    impostor:              {impostor_count}")
    print("  recording_split.csv:     recording-level audit")

    print(f"\nSaved to: {OUTPUT_DIR}")


# ======================== MAIN ========================

def main() -> None:
    rng = random.Random(SEED)

    speakers = scan_dataset()
    training_speakers, verification_speakers = choose_speakers(
        speakers,
        rng,
    )

    train_rows, dev_rows, train_dev_audit = split_train_dev(
        speakers,
        training_speakers,
        rng,
    )

    enrol_rows, test_rows, verification_audit = split_enrol_test(
        speakers,
        verification_speakers,
        rng,
    )

    recording_split_rows = train_dev_audit + verification_audit

    trials = create_trials(
        enrol_rows,
        test_rows,
        rng,
    )

    validate_split(
        train_rows,
        dev_rows,
        enrol_rows,
        test_rows,
    )

    # Delete old E0 manifests only after all new data passes validation.
    reset_output_dir()

    write_csv(OUTPUT_DIR / "train.csv", train_rows)
    write_csv(OUTPUT_DIR / "dev.csv", dev_rows)
    write_csv(OUTPUT_DIR / "enrol.csv", enrol_rows)
    write_csv(OUTPUT_DIR / "test.csv", test_rows)
    write_trials(
        OUTPUT_DIR / "verification_trials.txt",
        trials,
    )
    write_recording_split(
        OUTPUT_DIR / "recording_split.csv",
        recording_split_rows,
    )

    print_summary(
        speakers,
        training_speakers,
        verification_speakers,
        train_rows,
        dev_rows,
        enrol_rows,
        test_rows,
        recording_split_rows,
        trials,
    )


if __name__ == "__main__":
    main()