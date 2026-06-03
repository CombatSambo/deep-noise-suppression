from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
import yaml

from data_pipeline import build_train_dataloader, build_train_dataset_from_config_dict
from models.baseline_pytorch import create_mask_model


def _format_seconds(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _build_features_and_targets(
    noisy: torch.Tensor,
    clean: torch.Tensor,
    frame_size: int = 320,
    hop_size: int = 160,
    dft_size: int = 320,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build model features and training target mask.

    Input tensors are [B, 1, T], output tensors are:
    - features: [num_frames, B, 322]
    - target_mask: [num_frames, B, 161]
    """
    noisy = noisy.squeeze(1)
    clean = clean.squeeze(1)

    # Match enhancement frontend behavior: prepend one hop of zeros.
    noisy = F.pad(noisy, (hop_size, 0))
    clean = F.pad(clean, (hop_size, 0))

    window = torch.sqrt(torch.hann_window(frame_size, periodic=True, device=noisy.device))

    noisy_stft = torch.stft(
        noisy,
        n_fft=dft_size,
        hop_length=hop_size,
        win_length=frame_size,
        window=window,
        center=False,
        return_complex=True,
    )
    clean_stft = torch.stft(
        clean,
        n_fft=dft_size,
        hop_length=hop_size,
        win_length=frame_size,
        window=window,
        center=False,
        return_complex=True,
    )

    # Baseline enhancement loop runs one frame fewer than raw STFT frame count.
    if noisy_stft.shape[-1] > 1:
        noisy_stft = noisy_stft[..., :-1]
        clean_stft = clean_stft[..., :-1]

    noisy_mag = noisy_stft.abs().clamp_min(1e-12)
    clean_mag = clean_stft.abs().clamp_min(1e-12)

    feat_mic = torch.log10(noisy_mag.pow(2)).div(20.0)
    # No far-end signal in current training data; keep second feature branch neutral.
    feat_far = torch.full_like(feat_mic, -0.6)

    feat = torch.cat([feat_mic, feat_far], dim=1)  # [B, 322, T_frames]
    feat = feat.permute(2, 0, 1).contiguous()  # [T_frames, B, 322]

    target_mask = torch.clamp(clean_mag / noisy_mag, 0.0, 1.0)  # [B, 161, T_frames]
    target_mask = target_mask.permute(2, 0, 1).contiguous()  # [T_frames, B, 161]

    return feat, target_mask


def _run_eval_loop(model: torch.nn.Module, dataloader, device: torch.device) -> float:
    model.eval()
    running_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in dataloader:
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)

            features, target_mask = _build_features_and_targets(noisy=noisy, clean=clean)
            pred_mask, _, _ = model(features)
            loss = F.mse_loss(pred_mask, target_mask)

            running_loss += float(loss.item())
            steps += 1
    return running_loss / max(1, steps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PyTorch baseline speech-enhancement model.")
    parser.add_argument("--config", type=str, default="synthesizer_config.yaml")
    parser.add_argument(
        "--model-type",
        type=str,
        default="gru",
        choices=["gru", "encoder_decoder"],
        help="Model architecture: recurrent GRU baseline or recurrent GRU encoder-decoder mask network.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epoch-size", type=int, default=2000)
    parser.add_argument(
        "--val-epoch-size",
        type=int,
        default=0,
        help="Validation synthetic examples per epoch. If 0, uses max(1, epoch_size//10).",
    )
    parser.add_argument(
        "--test-epoch-size",
        type=int,
        default=0,
        help="Test synthetic examples for final test loop. If 0, uses val_epoch_size when file split is enabled.",
    )
    parser.add_argument("--segment-seconds", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint to resume from.")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--enable-file-split", action="store_true", help="Enable strict file-level train/val/test split.")
    parser.add_argument("--split-train-ratio", type=float, default=0.8)
    parser.add_argument("--split-val-ratio", type=float, default=0.1)
    parser.add_argument("--split-test-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("train_data", {})
    cfg.setdefault("train_dataloader", {})
    val_epoch_size = int(args.val_epoch_size) if int(args.val_epoch_size) > 0 else max(1, int(args.epoch_size) // 10)
    test_epoch_size = int(args.test_epoch_size) if int(args.test_epoch_size) > 0 else val_epoch_size

    train_cfg = dict(cfg)
    train_cfg["train_data"] = dict(cfg["train_data"])
    train_cfg["train_dataloader"] = dict(cfg["train_dataloader"])
    train_cfg["train_data"]["segment_seconds"] = float(args.segment_seconds)
    train_cfg["train_data"]["epoch_size"] = int(args.epoch_size)
    train_cfg["train_dataloader"]["batch_size"] = int(args.batch_size)
    train_cfg["train_dataloader"]["num_workers"] = int(args.num_workers)
    train_cfg["train_dataloader"]["shuffle"] = True
    train_cfg["train_dataloader"]["drop_last"] = False
    train_cfg["train_dataloader"]["pin_memory"] = False
    if args.enable_file_split:
        train_cfg["train_data"]["split"] = {
            "enabled": True,
            "train_ratio": float(args.split_train_ratio),
            "val_ratio": float(args.split_val_ratio),
            "test_ratio": float(args.split_test_ratio),
            "seed": int(args.split_seed),
        }

    val_cfg = dict(cfg)
    val_cfg["train_data"] = dict(cfg["train_data"])
    val_cfg["train_dataloader"] = dict(cfg["train_dataloader"])
    val_cfg["train_data"]["segment_seconds"] = float(args.segment_seconds)
    val_cfg["train_data"]["epoch_size"] = val_epoch_size
    val_cfg["train_dataloader"]["batch_size"] = int(args.batch_size)
    val_cfg["train_dataloader"]["num_workers"] = int(args.num_workers)
    val_cfg["train_dataloader"]["shuffle"] = False
    val_cfg["train_dataloader"]["drop_last"] = False
    val_cfg["train_dataloader"]["pin_memory"] = False
    if args.enable_file_split:
        val_cfg["train_data"]["split"] = {
            "enabled": True,
            "train_ratio": float(args.split_train_ratio),
            "val_ratio": float(args.split_val_ratio),
            "test_ratio": float(args.split_test_ratio),
            "seed": int(args.split_seed),
        }

    test_cfg = dict(cfg)
    test_cfg["train_data"] = dict(cfg["train_data"])
    test_cfg["train_dataloader"] = dict(cfg["train_dataloader"])
    test_cfg["train_data"]["segment_seconds"] = float(args.segment_seconds)
    test_cfg["train_data"]["epoch_size"] = test_epoch_size
    test_cfg["train_dataloader"]["batch_size"] = int(args.batch_size)
    test_cfg["train_dataloader"]["num_workers"] = int(args.num_workers)
    test_cfg["train_dataloader"]["shuffle"] = False
    test_cfg["train_dataloader"]["drop_last"] = False
    test_cfg["train_dataloader"]["pin_memory"] = False
    if args.enable_file_split:
        test_cfg["train_data"]["split"] = {
            "enabled": True,
            "train_ratio": float(args.split_train_ratio),
            "val_ratio": float(args.split_val_ratio),
            "test_ratio": float(args.split_test_ratio),
            "seed": int(args.split_seed),
        }

    train_dataset = build_train_dataset_from_config_dict(
        cfg=train_cfg,
        seed=args.seed,
        dataset_split="train" if args.enable_file_split else None,
    )
    train_dataloader = build_train_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=False,
        pin_memory=False,
        seed=args.seed,
    )
    val_dataset = build_train_dataset_from_config_dict(
        cfg=val_cfg,
        seed=args.seed + 10_000,
        dataset_split="val" if args.enable_file_split else None,
    )
    val_dataloader = build_train_dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
        seed=args.seed + 20_000,
    )
    test_dataloader = None
    if args.enable_file_split:
        test_dataset = build_train_dataset_from_config_dict(
            cfg=test_cfg,
            seed=args.seed + 30_000,
            dataset_split="test",
        )
        test_dataloader = build_train_dataloader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
            seed=args.seed + 40_000,
        )

    if args.device.lower() == "cuda" and not torch.cuda.is_available():
        print("[train][warning] CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device.lower())

    model = create_mask_model(args.model_type).to(device)
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        state = torch.load(str(resume_path), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            ckpt_model_type = state.get("model_type")
            if ckpt_model_type is not None and ckpt_model_type != args.model_type:
                raise ValueError(
                    f"Checkpoint model_type='{ckpt_model_type}' does not match --model-type='{args.model_type}'."
                )
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"[train] resumed from checkpoint: {resume_path}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[train] model_type={args.model_type} device={device} epochs={args.epochs} batch_size={args.batch_size} "
        f"epoch_size={args.epoch_size} val_epoch_size={val_epoch_size} test_epoch_size={test_epoch_size} "
        f"segment_seconds={args.segment_seconds}"
    )
    if args.enable_file_split:
        print(
            "[train] strict file split enabled: "
            f"train={args.split_train_ratio:.3f} val={args.split_val_ratio:.3f} "
            f"test={args.split_test_ratio:.3f} seed={args.split_seed}"
        )

    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        num_steps = 0

        for batch in train_dataloader:
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)

            features, target_mask = _build_features_and_targets(noisy=noisy, clean=clean)
            pred_mask, _, _ = model(features)

            loss = F.mse_loss(pred_mask, target_mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running_loss += float(loss.item())
            num_steps += 1

        train_loss = running_loss / max(1, num_steps)

        val_loss = _run_eval_loop(model=model, dataloader=val_dataloader, device=device)
        epoch_seconds = time.perf_counter() - epoch_start
        elapsed_total = time.perf_counter() - train_start
        avg_epoch_seconds = elapsed_total / epoch
        remaining_epochs = args.epochs - epoch
        eta_seconds = avg_epoch_seconds * remaining_epochs

        epoch_ckpt = save_dir / f"baseline_pytorch_epoch_{epoch:03d}.pt"
        torch.save(
            {
                "model_type": args.model_type,
                "state_dict": model.state_dict(),
            },
            epoch_ckpt,
        )
        print(
            f"[train] epoch={epoch}/{args.epochs} time={epoch_seconds:.2f}s "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} eta={_format_seconds(eta_seconds)}"
        )

    last_ckpt = save_dir / "baseline_pytorch_last.pt"
    torch.save(
        {
            "model_type": args.model_type,
            "state_dict": model.state_dict(),
        },
        last_ckpt,
    )
    print(f"[train] final checkpoint: '{last_ckpt}'")
    if test_dataloader is not None:
        test_loss = _run_eval_loop(model=model, dataloader=test_dataloader, device=device)
        print(f"[train] test_loss={test_loss:.6f}")


if __name__ == "__main__":
    main()
