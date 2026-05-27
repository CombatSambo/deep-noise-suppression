#!/bin/bash
#SBATCH -J dns_train
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
mkdir -p logs checkpoints_tesla_3h outputs

singularity exec --nv -B /gpfs:/gpfs "$CONTAINER" bash -lc '
  cd /gpfs/helios/home/etais/hpc_bacha/deep-noise-suppression
  source .venv/bin/activate

  echo "GPU check:"
  nvidia-smi

  python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); x=torch.randn((1024,1024),device=\"cuda\"); y=x@x; torch.cuda.synchronize(); print(\"cuda matmul ok\", y.mean().item())"

  python train_baseline_pytorch.py \
    --config synthesizer_config.yaml \
    --device cuda \
    --epochs 2 \
    --batch-size 32 \
    --epoch-size 180000 \
    --val-epoch-size 18000 \
    --segment-seconds 1.5 \
    --num-workers 8 \
    --lr 5e-4 \
    --grad-clip 5.0 \
    --save-dir checkpoints_tesla_3h
'