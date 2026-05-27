from pathlib import Path
import argparse
import soundfile as sf

from synthesizer import Synthesizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="synthesizer_config.yaml")
    parser.add_argument("--output-dir", default="outputs/synth_eval")
    parser.add_argument("--num-samples", type=int, default=50)
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    noisy_dir = out_root / "noisy"
    target_dir = out_root / "target"
    nearend_dir = out_root / "nearend"

    noisy_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    nearend_dir.mkdir(parents=True, exist_ok=True)

    synth = Synthesizer(args.config)
    sr = synth.nearend_datasets.sample_rate

    for i in range(args.num_samples):
        audio = synth.generate()
        stem = f"sample_{i:04d}"

        sf.write(noisy_dir / f"{stem}_mic.wav", audio["mic"], sr)
        sf.write(target_dir / f"{stem}_target.wav", audio["target"], sr)
        sf.write(nearend_dir / f"{stem}_nearend.wav", audio["nearend"], sr)

        print(f"[generate] {stem}")

    print(f"[generate] done: {args.num_samples} samples")
    print(f"[generate] noisy: {noisy_dir}")
    print(f"[generate] target: {target_dir}")
    print(f"[generate] nearend: {nearend_dir}")


if __name__ == "__main__":
    main()