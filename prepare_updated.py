"""Prepare data for a minimal SpeechBrain ECAPA-TDNN baseline.

Expected structure:
DATASET_ROOT/
    speaker_01/*.wav
    speaker_02/*.wav
    ...

All audio files must be mono, 16 kHz, and exactly 3 seconds.
"""

import csv
import random
import re
import shutil
from pathlib import Path

import soundfile as sf


# ======================== CONFIG ========================

PROJECT_ROOT = Path("/Users/tavantai/Developer/project_thesis_code")
DATASET_ROOT = PROJECT_ROOT / "dataset_copy"
OUTPUT_DIR = PROJECT_ROOT / "manifests"

SAMPLE_RATE = 16000
DURATION = 3.0
DEV_RATIO = 0.10

NUM_VERIFICATION_SPEAKERS = 2
ENROL_FILES_PER_SPEAKER = 5

# None: choose automatically with SEED.
# Or specify names, for example: ["speaker_06", "speaker_07"]
VERIFICATION_SPEAKERS = None

SEED = 42
AUDIO_EXTENSIONS = {".wav", ".flac"}


# ======================== HELPERS ========================

def reset_output_dir() -> None:
    """Delete the old manifests folder and create a new empty one."""
    if OUTPUT_DIR.name != "manifests":
        raise ValueError(
            "Safety check failed: OUTPUT_DIR must point to "
            "a folder named 'manifests'."
        )

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)


def make_id(path: Path) -> str:
    """Create a unique ID from the path relative to DATASET_ROOT."""
    text = "--".join(path.relative_to(DATASET_ROOT).with_suffix("").parts)
    return re.sub(r"[^A-Za-z0-9_-]", "_", text)


def read_audio(path: Path) -> dict:
    """Validate one 3-second audio file and return its manifest row data."""
    info = sf.info(str(path))
    expected_frames = int(SAMPLE_RATE * DURATION)

    if info.samplerate != SAMPLE_RATE:
        raise ValueError(
            f"{path}: expected {SAMPLE_RATE} Hz, got {info.samplerate} Hz"
        )

    if info.channels != 1:
        raise ValueError(
            f"{path}: expected mono audio, got {info.channels} channels"
        )

    if info.frames != expected_frames:
        raise ValueError(
            f"{path}: expected {expected_frames} samples, got {info.frames}"
        )

    return {
        "ID": make_id(path),
        "duration": DURATION,
        "wav": str(path.resolve()),
        "start": 0,
        "stop": expected_frames,
        "spk_id": path.relative_to(DATASET_ROOT).parts[0],
    }


def scan_dataset() -> dict[str, list[dict]]:
    """Read all speakers and audio files."""
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Dataset not found: {DATASET_ROOT}")

    speakers = {}

    for speaker_dir in sorted(DATASET_ROOT.iterdir()):
        if not speaker_dir.is_dir():
            continue

        paths = sorted(
            path
            for path in speaker_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        )

        if paths:
            speakers[speaker_dir.name] = [read_audio(path) for path in paths]

    if len(speakers) <= NUM_VERIFICATION_SPEAKERS:
        raise ValueError("Not enough speakers for training and verification.")

    return speakers


def choose_speakers(
    speakers: dict[str, list[dict]],
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    """Create disjoint training and verification speaker groups."""
    min_verification_files = ENROL_FILES_PER_SPEAKER + 1

    if VERIFICATION_SPEAKERS is None:
        eligible = [
            spk
            for spk, rows in speakers.items()
            if len(rows) >= min_verification_files
        ]

        if len(eligible) < NUM_VERIFICATION_SPEAKERS:
            raise ValueError("Not enough eligible verification speakers.")

        verification = sorted(
            rng.sample(eligible, NUM_VERIFICATION_SPEAKERS)
        )
    else:
        verification = sorted(VERIFICATION_SPEAKERS)

        if len(verification) != NUM_VERIFICATION_SPEAKERS:
            raise ValueError(
                "VERIFICATION_SPEAKERS length must match "
                "NUM_VERIFICATION_SPEAKERS."
            )

        for spk in verification:
            if spk not in speakers:
                raise ValueError(f"Unknown verification speaker: {spk}")

            if len(speakers[spk]) < min_verification_files:
                raise ValueError(
                    f"{spk} needs at least {min_verification_files} files."
                )

    training = sorted(set(speakers) - set(verification))
    return training, verification


def split_train_dev(
    speakers: dict[str, list[dict]],
    training_speakers: list[str],
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    """Split files of every training speaker into train and dev."""
    train_rows = []
    dev_rows = []

    for spk in training_speakers:
        rows = list(speakers[spk])

        if len(rows) < 2:
            raise ValueError(f"{spk} needs at least 2 files for train/dev.")

        rng.shuffle(rows)

        num_dev = max(1, round(len(rows) * DEV_RATIO))
        num_dev = min(num_dev, len(rows) - 1)

        dev_rows.extend(rows[:num_dev])
        train_rows.extend(rows[num_dev:])

    return train_rows, dev_rows


def split_enrol_test(
    speakers: dict[str, list[dict]],
    verification_speakers: list[str],
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    """Use the same number of files for every verification speaker."""
    balanced_count = min(
        len(speakers[spk]) for spk in verification_speakers
    )

    if balanced_count <= ENROL_FILES_PER_SPEAKER:
        raise ValueError("Not enough files to create enrol and test sets.")

    enrol_rows = []
    test_rows = []

    for spk in verification_speakers:
        rows = list(speakers[spk])
        rng.shuffle(rows)
        rows = rows[:balanced_count]

        enrol_rows.extend(rows[:ENROL_FILES_PER_SPEAKER])
        test_rows.extend(rows[ENROL_FILES_PER_SPEAKER:])

    return enrol_rows, test_rows


def create_trials(
    enrol_rows: list[dict],
    test_rows: list[dict],
    rng: random.Random,
) -> list[tuple[int, str, str]]:
    """Create balanced genuine and impostor trials."""
    genuine = []
    impostor = []

    for enrol in enrol_rows:
        for test in test_rows:
            label = int(enrol["spk_id"] == test["spk_id"])
            trial = (label, enrol["ID"], test["ID"])

            if label == 1:
                genuine.append(trial)
            else:
                impostor.append(trial)

    if not genuine or not impostor:
        raise ValueError("Trials need both genuine and impostor pairs.")

    rng.shuffle(impostor)
    impostor = impostor[:len(genuine)]

    trials = genuine + impostor
    rng.shuffle(trials)

    return trials


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write one SpeechBrain manifest CSV."""
    columns = ["ID", "duration", "wav", "start", "stop", "spk_id"]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_trials(
    path: Path,
    trials: list[tuple[int, str, str]],
) -> None:
    """Write verification pairs."""
    with path.open("w", encoding="utf-8") as file:
        for label, enrol_id, test_id in trials:
            file.write(f"{label} {enrol_id} {test_id}\n")


def validate_split(
    train_rows: list[dict],
    dev_rows: list[dict],
    enrol_rows: list[dict],
    test_rows: list[dict],
) -> None:
    """Check speaker separation and duplicate IDs."""
    train_spks = {row["spk_id"] for row in train_rows}
    dev_spks = {row["spk_id"] for row in dev_rows}
    verification_spks = {
        row["spk_id"] for row in enrol_rows + test_rows
    }

    if train_spks != dev_spks:
        raise ValueError("train.csv and dev.csv must have the same speakers.")

    if not train_spks.isdisjoint(verification_spks):
        raise ValueError("Training and verification speakers overlap.")

    all_rows = train_rows + dev_rows + enrol_rows + test_rows
    ids = [row["ID"] for row in all_rows]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate audio IDs found across splits.")


# ======================== MAIN ========================

def main() -> None:
    rng = random.Random(SEED)

    speakers = scan_dataset()
    training_spks, verification_spks = choose_speakers(speakers, rng)

    train_rows, dev_rows = split_train_dev(
        speakers,
        training_spks,
        rng,
    )

    enrol_rows, test_rows = split_enrol_test(
        speakers,
        verification_spks,
        rng,
    )

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

    # Delete the old manifests folder only after validation succeeds.
    reset_output_dir()

    write_csv(OUTPUT_DIR / "train.csv", train_rows)
    write_csv(OUTPUT_DIR / "dev.csv", dev_rows)
    write_csv(OUTPUT_DIR / "enrol.csv", enrol_rows)
    write_csv(OUTPUT_DIR / "test.csv", test_rows)
    write_trials(
        OUTPUT_DIR / "verification_trials.txt",
        trials,
    )

    print("\nTraining speakers:")
    print(" ", training_spks)

    print("\nVerification speakers:")
    print(" ", verification_spks)

    print("\nGenerated manifests:")
    print(f"  train.csv: {len(train_rows)} files")
    print(f"  dev.csv: {len(dev_rows)} files")
    print(f"  enrol.csv: {len(enrol_rows)} files")
    print(f"  test.csv: {len(test_rows)} files")
    print(f"  verification_trials.txt: {len(trials)} pairs")

    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
