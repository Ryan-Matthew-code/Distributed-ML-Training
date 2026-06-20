# Distributed ML Training Pipeline

A reference implementation of distributed model training using PyTorch's
`DistributedDataParallel` (DDP), designed to demonstrate production-grade
patterns for scaling training across multiple GPUs and nodes.

## Why this project

Training large models efficiently requires more than writing a training
loop — it requires correctly sharding data across workers, synchronizing
gradients, managing checkpoints safely, and making the whole pipeline
portable across environments (local GPU box, Kubernetes cluster, cloud VM).
This repo packages those patterns into a small, readable codebase.

## Key features

- **DDP-based multi-GPU/multi-node training** launched via `torchrun`,
  avoiding manual process/rank bookkeeping.
- **DistributedSampler** for correct, non-overlapping data sharding across
  ranks each epoch.
- **Mixed-precision training** (`torch.autocast` + `GradScaler`) for
  improved throughput on supported hardware.
- **Rank-0-only checkpointing** to avoid race conditions when multiple
  processes write to disk simultaneously.
- **Containerized** with a Dockerfile for reproducible environments.
- **Kubernetes Job manifest** showing how this would be scheduled in a
  cluster setting.

## Project structure

```
project1-distributed-ml-training/
├── src/
│   ├── train.py     # Main distributed training loop
│   ├── model.py     # Model definition (SimpleCNN)
│   └── data.py      # Dataset loading utility (CIFAR-10 by default)
├── k8s/
│   └── training-job.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Running locally (single machine, multiple GPUs)

```bash
pip install -r requirements.txt
torchrun --nproc_per_node=<NUM_GPUS> src/train.py --epochs 10 --batch-size 64
```

If no GPU is available, the script falls back to CPU/gloo backend
automatically (useful for testing the pipeline logic).

## Running on Kubernetes

```bash
docker build -t <your-registry>/distributed-ml-training:latest .
docker push <your-registry>/distributed-ml-training:latest
kubectl apply -f k8s/training-job.yaml
```

## Adapting this to your own data/model

- Swap `get_dataset()` in `src/data.py` for your own `torch.utils.data.Dataset`.
- Swap `SimpleCNN` in `src/model.py` for your architecture of choice.
- The training loop, checkpointing, and distributed setup require no changes.

## Roadmap

- [ ] Add multi-node launch example (`torchrun --nnodes=2 ...`)
- [ ] Integrate experiment tracking (see companion MLOps Pipeline project)
- [ ] Add gradient accumulation for large effective batch sizes
- [ ] Add fault-tolerant/elastic training example
