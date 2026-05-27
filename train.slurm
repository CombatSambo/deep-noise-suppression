#!/bin/bash
#SBATCH -J dns_v100_10ep
#SBATCH -A ealloc_ati-1-neur
#SBATCH -p gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_DIR=/gpfs/helios/home/etais/hpc_bacha/deep-noise-suppression
CONTAINER=/gpfs/helios/projects/neuralnet_course/track02/containers/pytorch_26.04-py3.sif

cd "$PROJECT_DIR"

module load singularity

mkdir -p logs checkpoints_v100_10ep outputs

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
    --device cuda \
    --epochs 10 \
    --batch-size 32 \
    --epoch-size 50000 \
    --val-epoch-size 5000 \
    --segment-seconds 1.0 \
    --num-workers 8 \
    --lr 5e-4 \
    --grad-clip 5.0 \
    --save-dir checkpoints_v100_10ep

  echo "===== Training finished ====="
  ls -lh checkpoints_v100_10ep
'