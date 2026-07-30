#!/usr/bin/env python3
"""Preprocess raw recordings for the ECAPA-TDNN baseline E0.

Operations performed:
- Recursively scan the raw dataset.
- Convert multi-channel audio to mono by averaging channels.
- Resample audio to 16 kHz.
- Preserve recording duration and relative folder structure.
- Save processed recordings as PCM-16 WAV files.

Operations intentionally NOT performed:
- VAD
- Noise/reverberation augmentation
- Chunking
- Padding
- Train/dev/enrol/test splitting

Default structure:
    /Users/tavantai/Developer/project_thesis_code/dataset_raw/
        speaker_01/*.wav
        speaker_02/*.wav

Output:
    /Users/tavantai/Developer/project_thesis_code/processed/E0/
        speaker_01/*.wav
        speaker_02/*.wav
        preprocess_report.csv
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as AF


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "new_data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "processed"
DEFAULT_SAMPLE_RATE = 16_000
SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


@dataclass
class ProcessResult:
    input_path: str
    output_path: str
    status: str
    original_sample_rate: int | str = ""
    original_channels: int | str = ""
    original_frames: int | str = ""
    output_sample_rate: int | str = ""
    output_channels: int | str = ""
    output_frames: int | str = ""
    original_duration_seconds: float | str = ""
    output_duration_seconds: float | str = ""
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw recordings to mono 16 kHz WAV files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Raw dataset directory. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Processed dataset directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Target sample rate. Default: {DEFAULT_SAMPLE_RATE}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite processed files that already exist.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete the output directory before processing.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when one file fails.",
    )
    return parser.parse_args()


def validate_paths(input_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    if input_dir == output_dir:
        raise ValueError("Input and output directories must be different.")

    try:
        output_dir.relative_to(input_dir)
    except ValueError:
        pass
    else:
        raise ValueError(
            "Output directory must not be inside the input directory: "
            f"{output_dir}"
        )

    return input_dir, output_dir


def safely_clean_output(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    if len(output_dir.parts) < 4:
        raise ValueError(f"Refusing to delete a broad path: {output_dir}")
    shutil.rmtree(output_dir)


def discover_audio_files(input_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in input_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: str(path).lower(),
    )


def get_output_path(
    input_path: Path,
    input_dir: Path,
    output_dir: Path,
) -> Path:
    relative_path = input_path.relative_to(input_dir)
    return (output_dir / relative_path).with_suffix(".wav")


def load_audio(input_path: Path) -> tuple[torch.Tensor, int, int, int]:
    """Load audio as a float32 tensor shaped [channels, frames]."""
    audio, sample_rate = sf.read(
        input_path,
        dtype="float32",
        always_2d=True,
    )

    frames, channels = audio.shape
    if frames == 0:
        raise ValueError("Audio contains zero frames.")
    if channels == 0:
        raise ValueError("Audio contains zero channels.")

    waveform = torch.from_numpy(audio.T.copy())
    if not torch.isfinite(waveform).all():
        raise ValueError("Audio contains NaN or infinite values.")

    return waveform, int(sample_rate), int(channels), int(frames)


def convert_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim != 2:
        raise ValueError(
            "Expected waveform with shape [channels, frames], "
            f"got {tuple(waveform.shape)}"
        )
    if waveform.shape[0] == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


def resample_audio(
    waveform: torch.Tensor,
    original_sample_rate: int,
    target_sample_rate: int,
) -> torch.Tensor:
    if original_sample_rate <= 0:
        raise ValueError(f"Invalid sample rate: {original_sample_rate}")
    if target_sample_rate <= 0:
        raise ValueError(f"Invalid target rate: {target_sample_rate}")
    if original_sample_rate == target_sample_rate:
        return waveform

    return AF.resample(
        waveform=waveform,
        orig_freq=original_sample_rate,
        new_freq=target_sample_rate,
    )


def save_mono_wav(
    output_path: Path,
    waveform: torch.Tensor,
    sample_rate: int,
) -> None:
    if waveform.ndim != 2 or waveform.shape[0] != 1:
        raise ValueError(
            "Processed waveform must have shape [1, frames], "
            f"got {tuple(waveform.shape)}"
        )

    waveform = torch.nan_to_num(
        waveform,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    ).clamp(-1.0, 1.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        output_path,
        waveform.squeeze(0).cpu().numpy(),
        sample_rate,
        format="WAV",
        subtype="PCM_16",
    )


def verify_output(
    output_path: Path,
    target_sample_rate: int,
    original_duration: float,
) -> sf.SoundFile:
    info = sf.info(output_path)

    if info.samplerate != target_sample_rate:
        raise ValueError(
            f"Expected {target_sample_rate} Hz, got {info.samplerate} Hz"
        )
    if info.channels != 1:
        raise ValueError(f"Expected mono audio, got {info.channels} channels")

    output_duration = info.frames / info.samplerate
    tolerance = max(2.0 / target_sample_rate, 1e-4)
    if abs(output_duration - original_duration) > tolerance:
        raise ValueError(
            "Duration changed unexpectedly: "
            f"before={original_duration:.6f}s, "
            f"after={output_duration:.6f}s"
        )

    return info


def process_one_file(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int,
    overwrite: bool,
) -> ProcessResult:
    if output_path.exists() and not overwrite:
        return ProcessResult(
            input_path=str(input_path),
            output_path=str(output_path),
            status="skipped_existing",
        )

    waveform, original_sr, channels, original_frames = load_audio(input_path)
    original_duration = original_frames / original_sr

    waveform = convert_to_mono(waveform)
    waveform = resample_audio(
        waveform,
        original_sample_rate=original_sr,
        target_sample_rate=target_sample_rate,
    )
    save_mono_wav(output_path, waveform, target_sample_rate)

    output_info = verify_output(
        output_path,
        target_sample_rate,
        original_duration,
    )

    return ProcessResult(
        input_path=str(input_path),
        output_path=str(output_path),
        status="processed",
        original_sample_rate=original_sr,
        original_channels=channels,
        original_frames=original_frames,
        output_sample_rate=output_info.samplerate,
        output_channels=output_info.channels,
        output_frames=output_info.frames,
        original_duration_seconds=round(original_duration, 6),
        output_duration_seconds=round(
            output_info.frames / output_info.samplerate,
            6,
        ),
    )


def write_report(results: list[ProcessResult], output_dir: Path) -> Path:
    report_path = output_dir / "preprocess_report.csv"
    fieldnames = list(ProcessResult.__dataclass_fields__.keys())

    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)

    return report_path


def main() -> int:
    args = parse_args()

    try:
        input_dir, output_dir = validate_paths(
            args.input_dir,
            args.output_dir,
        )

        if args.clean_output:
            safely_clean_output(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_files = discover_audio_files(input_dir)
        if not audio_files:
            extensions = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise FileNotFoundError(
                f"No supported audio files found in {input_dir}. "
                f"Supported extensions: {extensions}"
            )

        print(f"Input directory : {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Target rate     : {args.sample_rate} Hz")
        print(f"Audio files     : {len(audio_files)}")

        results: list[ProcessResult] = []
        claimed_outputs: dict[Path, Path] = {}

        for index, input_path in enumerate(audio_files, start=1):
            output_path = get_output_path(
                input_path,
                input_dir,
                output_dir,
            )

            previous_input = claimed_outputs.get(output_path)
            if previous_input is not None and previous_input != input_path:
                error = (
                    f"Output collision: {previous_input} and {input_path} "
                    f"both map to {output_path}"
                )
                results.append(
                    ProcessResult(
                        input_path=str(input_path),
                        output_path=str(output_path),
                        status="failed",
                        error=error,
                    )
                )
                print(f"[{index}/{len(audio_files)}] FAILED: {error}")
                if args.fail_fast:
                    break
                continue

            claimed_outputs[output_path] = input_path

            try:
                result = process_one_file(
                    input_path,
                    output_path,
                    args.sample_rate,
                    args.overwrite,
                )
                results.append(result)
                print(
                    f"[{index}/{len(audio_files)}] {result.status}: "
                    f"{input_path.relative_to(input_dir)}"
                )
            except Exception as exc:
                results.append(
                    ProcessResult(
                        input_path=str(input_path),
                        output_path=str(output_path),
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                print(
                    f"[{index}/{len(audio_files)}] FAILED: "
                    f"{input_path.relative_to(input_dir)} — {exc}",
                    file=sys.stderr,
                )
                if args.fail_fast:
                    break

        report_path = write_report(results, output_dir)

        processed = sum(item.status == "processed" for item in results)
        skipped = sum(item.status == "skipped_existing" for item in results)
        failed = sum(item.status == "failed" for item in results)

        print("\nPreprocessing summary")
        print("---------------------")
        print(f"Processed : {processed}")
        print(f"Skipped   : {skipped}")
        print(f"Failed    : {failed}")
        print(f"Report    : {report_path}")

        return 1 if failed else 0

    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
