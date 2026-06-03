from __future__ import annotations

import argparse
import csv
import copy
from pathlib import Path
from typing import Any, Dict

import soundfile as sf
import yaml

from data_pipeline import build_train_dataset_from_config_dict


def _prepare_config(args: argparse.Namespace) -> Dict[str, Any]:
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = copy.deepcopy(cfg)
    cfg.setdefault("train_data", {})
    cfg["train_data"]["epoch_size"] = int(args.num_samples)
    if args.segment_seconds is not None:
        cfg["train_data"]["segment_seconds"] = float(args.segment_seconds)

    cfg["train_data"]["split"] = {
        "enabled": True,
        "train_ratio": float(args.split_train_ratio),
        "val_ratio": float(args.split_val_ratio),
        "test_ratio": float(args.split_test_ratio),
        "seed": int(args.split_seed),
    }
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a fixed eval set from a strict file-level split.")
    parser.add_argument("--config", type=str, default="synthesizer_config.yaml")
    parser.add_argument("--output-dir", type=str, default="outputs/split_test_eval")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--split-train-ratio", type=float, default=0.8)
    parser.add_argument("--split-val-ratio", type=float, default=0.1)
    parser.add_argument("--split-test-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--segment-seconds", type=float, default=None)
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be > 0")

    cfg = _prepare_config(args)
    dataset = build_train_dataset_from_config_dict(cfg=cfg, seed=args.seed, dataset_split=args.split)

    out_root = Path(args.output_dir)
    noisy_dir = out_root / "noisy"
    target_dir = out_root / "target"
    noise_dir = out_root / "noise"
    noisy_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    noise_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = out_root / "metadata.csv"
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample",
                "split",
                "sample_rate",
                "snr_db",
                "clean_path",
                "noise_path",
                "clean_dataset",
                "noise_dataset",
            ],
        )
        writer.writeheader()

        for idx in range(args.num_samples):
            item = dataset[idx]
            sample_rate = int(item["sample_rate"].item())
            stem = f"sample_{idx:04d}"

            noisy = item["noisy"].squeeze(0).numpy()
            target = item["clean"].squeeze(0).numpy()
            noise = item["noise"].squeeze(0).numpy()

            sf.write(noisy_dir / f"{stem}_mic.wav", noisy, sample_rate)
            sf.write(target_dir / f"{stem}_target.wav", target, sample_rate)
            sf.write(noise_dir / f"{stem}_noise.wav", noise, sample_rate)

            writer.writerow(
                {
                    "sample": stem,
                    "split": args.split,
                    "sample_rate": sample_rate,
                    "snr_db": float(item["snr_db"].item()),
                    "clean_path": item["clean_path"],
                    "noise_path": item["noise_path"],
                    "clean_dataset": item["clean_dataset"],
                    "noise_dataset": item["noise_dataset"],
                }
            )
            print(f"[generate-split] {stem}")

    print(f"[generate-split] done: {args.num_samples} samples from split='{args.split}'")
    print(f"[generate-split] noisy: {noisy_dir}")
    print(f"[generate-split] target: {target_dir}")
    print(f"[generate-split] metadata: {metadata_path}")


if __name__ == "__main__":
    main()
