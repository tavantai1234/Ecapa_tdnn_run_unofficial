#!/usr/bin/env python3
"""Fine-tune a pretrained ECAPA-TDNN on MPS, CUDA, or CPU."""

import os

# Harmless outside macOS. On Apple MPS, unsupported operations may fall back
# to CPU instead of stopping the whole training process.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import csv
import sys

import speechbrain as sb
import torch
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio import audio_io
from speechbrain.utils.distributed import run_on_main


class SpeakerBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, lengths = batch.sig

        features = self.modules.compute_features(wavs)
        features = self.modules.mean_var_norm(features, lengths)
        embeddings = self.modules.embedding_model(features, lengths)
        predictions = self.modules.classifier(embeddings)

        return predictions, lengths

    def compute_objectives(self, predictions, batch, stage):
        predictions, lengths = predictions
        speaker_ids, _ = batch.spk_id_encoded

        loss = self.hparams.compute_cost(
            predictions,
            speaker_ids,
            lengths,
        )

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(
                batch.id,
                predictions,
                speaker_ids,
                lengths,
            )

        return loss

    def on_stage_start(self, stage, epoch=None):
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        stats = {"loss": stage_loss}

        if stage == sb.Stage.TRAIN:
            self.train_stats = stats
            return

        stats["ErrorRate"] = self.error_metrics.summarize("average")

        if stage == sb.Stage.VALID:
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr": self.optimizer.param_groups[0]["lr"],
                },
                train_stats=self.train_stats,
                valid_stats=stats,
            )

            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stats["ErrorRate"]},
                min_keys=["ErrorRate"],
            )


def dataio_prep(hparams):
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_annotation"]
    )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_annotation"]
    )
    datasets = [train_data, valid_data]

    @sb.utils.data_pipeline.takes("wav", "start", "stop")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop):
        start = int(start)
        stop = int(stop)

        signal, sample_rate = audio_io.load(
            wav,
            frame_offset=start,
            num_frames=stop - start,
        )

        if sample_rate != hparams["sample_rate"]:
            raise ValueError(
                f"{wav}: expected {hparams['sample_rate']} Hz, "
                f"got {sample_rate} Hz"
            )

        return signal.transpose(0, 1).squeeze(1)

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    @sb.utils.data_pipeline.takes("spk_id")
    @sb.utils.data_pipeline.provides("spk_id", "spk_id_encoded")
    def label_pipeline(spk_id):
        yield spk_id
        yield label_encoder.encode_sequence_torch([spk_id])

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    label_encoder.load_or_create(
        path=os.path.join(hparams["save_folder"], "label_encoder.txt"),
        from_didatasets=[train_data],
        output_key="spk_id",
    )

    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "spk_id_encoded"],
    )

    return train_data, valid_data


def count_speakers(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as file:
        return len({row["spk_id"] for row in csv.DictReader(file)})


def best_available_device():
    """Choose MPS first, then CUDA, and finally CPU."""
    mps_backend = getattr(torch.backends, "mps", None)

    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def set_default_device():
    """Add an automatic --device unless the user supplied one explicitly."""
    device_was_given = any(
        arg == "--device" or arg.startswith("--device=")
        for arg in sys.argv[1:]
    )

    if not device_was_given:
        sys.argv.append(f"--device={best_available_device()}")


def validate_device(device):
    device_name = str(device)

    if device_name.startswith("mps"):
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_built():
            raise RuntimeError(
                "This PyTorch installation was not built with MPS support."
            )
        if not mps_backend.is_available():
            raise RuntimeError(
                "MPS was requested but is unavailable."
            )

    elif device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but no CUDA GPU is available."
            )


def print_device_info(device):
    device_name = str(device)
    print(f"Selected device: {device_name}")

    if device_name.startswith("cuda"):
        cuda_index = torch.cuda.current_device()
        print(f"CUDA GPU: {torch.cuda.get_device_name(cuda_index)}")


if __name__ == "__main__":
    set_default_device()

    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    validate_device(run_opts["device"])
    print_device_info(run_opts["device"])

    with open(hparams_file, encoding="utf-8") as file:
        hparams = load_hyperpyyaml(file, overrides)

    os.makedirs(hparams["save_folder"], exist_ok=True)

    actual_speakers = count_speakers(hparams["train_annotation"])
    if actual_speakers != hparams["out_n_neurons"]:
        raise ValueError(
            f"train.csv has {actual_speakers} speakers, but "
            f"out_n_neurons={hparams['out_n_neurons']} in YAML."
        )

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    train_data, valid_data = dataio_prep(hparams)

    # Load only the pretrained ECAPA encoder.
    # The VoxCeleb classifier is intentionally not loaded.
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected()

    brain = SpeakerBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    brain.fit(
        epoch_counter=brain.hparams.epoch_counter,
        train_set=train_data,
        valid_set=valid_data,
        train_loader_kwargs=hparams["train_dataloader_options"],
        valid_loader_kwargs=hparams["valid_dataloader_options"],
    )