#!/bin/bash
#SBATCH -J dns_encdec_20ep_10s
#SBATCH -A ealloc_ati-1-neur
#SBATCH -p gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR=/gpfs/helios/home/etais/hpc_bacha/deep-noise-suppression
CONTAINER=/gpfs/helios/projects/neuralnet_course/track02/containers/pytorch_26.04-py3.sif

cd "$PROJECT_DIR"

module load singularity

mkdir -p logs checkpoints_encdec_20ep_10s outputs

singularity exec --nv -B /gpfs:/gpfs "$CONTAINER" bash -lc '
  cd /gpfs/helios/home/etais/hpc_bacha/deep-noise-suppression
  source .venv/bin/activate

  echo "===== GPU check ====="
  nvidia-smi

  echo "===== Torch check ====="
  python -c "import torch; print(\"torch:\", torch.__version__); print(\"cuda available:\", torch.cuda.is_available()); print(\"gpu:\", torch.cuda.get_device_name(0)); x=torch.randn((1024,1024),device=\"cuda\"); y=x@x; torch.cuda.synchronize(); print(\"cuda matmul ok:\", y.mean().item())"

  echo "===== Dataset check ====="
  ls -lah dataset
  ls dataset/clean_speech
  ls dataset/noise | head

  echo "===== Training start ====="
  python train_baseline_pytorch.py \
    --config synthesizer_config.yaml \
    --model-type encoder_decoder \
    --enable-file-split \
    --split-train-ratio 0.8 \
    --split-val-ratio 0.1 \
    --split-test-ratio 0.1 \
    --split-seed 42 \
    --device cuda \
    --epochs 20 \
    --batch-size 32 \
    --epoch-size 80000 \
    --val-epoch-size 8000 \
    --test-epoch-size 8000 \
    --segment-seconds 10.0 \
    --num-workers 8 \
    --lr 5e-4 \
    --grad-clip 5.0 \
    --save-dir checkpoints_encdec_20ep_10s

  echo "===== Training finished ====="
  ls -lh checkpoints_encdec_20ep_10s
'
