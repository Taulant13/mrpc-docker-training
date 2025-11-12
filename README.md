# MRPC Paraphrase Detection - Containerized Training Pipeline

A production-ready, containerized machine learning training pipeline for fine-tuning DistilBERT on the MRPC paraphrase detection task. This project demonstrates modern MLOps practices including Docker containerization, experiment tracking, and cloud deployment compatibility.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Cloud Deployment](#cloud-deployment)
- [Results](#results)

## Overview

This project tackles the paraphrase detection problem using the MRPC dataset from the GLUE benchmark. The model determines whether two sentences are semantic paraphrases of each other.

**Model Architecture:** DistilBERT (distilbert-base-uncased)  
**Task:** Binary classification (paraphrase vs. non-paraphrase)  
**Dataset:** MRPC (3,668 training pairs, 408 validation pairs)
**Framework:** PyTorch Lightning  
**Experiment Tracking:** Weights & Biases

### Key Achievements

- **85-86% accuracy** on MRPC validation set
- **89-90% F1 score**
- **Cloud-ready** with Docker containerization


## Quick Start

Start training in 5 minutes:
```bash
# 1. Clone the repository
git clone https://github.com/Taulant13/mrpc-docker-training.git
cd mrpc-docker-training

# 2. Build the Docker image
docker build -t mrpc-training .

# 3. Run training (with W&B logging)
docker run --rm -e WANDB_API_KEY=your_wandb_api_key mrpc-training
```
This will run with the best settings found on manual tuning: LR: **9e-5**, Warmup Steps: **150**, LR-Scheduler: **cosine**
Training should complete in 20-30 minutes on CPU.

```bash
# To change hyperparameters, W&B logging configurations
docker run --rm mrpc-training python train.py --help
```

## Installation

### Prerequisites

- **Docker Desktop** ([Download](https://www.docker.com/products/docker-desktop))
- **Weights & Biases Account** (free): [Sign up](https://wandb.ai/signup)
- **4GB+ RAM** available for Docker
- **~2GB disk space** for the Docker image

### Verify Docker Installation
```bash
docker --version
# Should output: Docker version 20.10.x or higher
```

### Get Your W&B API Key

1. Go to https://wandb.ai/authorize
2. Copy your API key (format: `a1b2c3d4e5f6...`)

## Usage

### Basic Training

Uses optimal hyperparameters discovered through manual tuning:
```bash
docker run --rm -e WANDB_API_KEY=your_api_key mrpc-training
```

**Expected output:**
- Training time: ~25-30 minutes (CPU)
- Final accuracy: ~85-86%
- Final F1 score: ~89-90%
- W&B dashboard link at completion


### Training Without W&B

For quick experiments without experiment tracking:
```bash
docker run --rm mrpc-training python train.py --no_wandb
```

### View All Available Options
```bash
docker run --rm mrpc-training python train.py --help
```

## Configuration

### Default Hyperparameters

Optimized through manual hyperparameter tuning:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `--lr` | 9e-5 | Learning rate |
| `--warmup_steps` | 150 | Learning rate warmup steps |
| `--scheduler_type` | cosine | LR scheduler (linear/cosine) |
| `--epochs` | 3 | Number of training epochs |
| `--train_batch_size` | 32 | Training batch size |
| `--eval_batch_size` | 32 | Evaluation batch size |
| `--weight_decay` | 0.0 | AdamW weight decay |
| `--seed` | 42 | Random seed for reproducibility |

### All Available Arguments

**Training Arguments:**
- `--learning_rate, --lr`: Learning rate
- `--warmup_steps`: Warmup steps for LR scheduler
- `--scheduler_type`: Scheduler type (linear, cosine)
- `--weight_decay`: Weight decay for regularization
- `--beta1`, `--beta2`: AdamW optimizer betas
- `--epochs`: Number of training epochs
- `--train_batch_size`: Training batch size
- `--eval_batch_size`: Evaluation batch size
- `--max_seq_length`: Maximum sequence length (default: 128)

**System Arguments:**
- `--seed`: Random seed for reproducibility
- `--accelerator`: Device type (auto, cpu, gpu, mps)
- `--devices`: Number of devices to use

**Experiment Tracking:**
- `--no_wandb`: Disable W&B logging
- `--wandb_project`: W&B project name
- `--wandb_run_name`: Custom run name (auto-generated if not provided)
- `--wandb_tags`: Tags for organizing runs
- `--wandb_api_key`: W&B API key

**Checkpointing:**
- `--checkpoint_dir`: Directory for saving model checkpoints (default: checkpoints)


## Cloud Deployment

### GitHub Codespaces

```bash
# 1. Go to your GitHub repository
# 2. Click "Code" → "Codespaces" → "Create codespace on main"
# 3. Wait for environment to load

# 4. In the Codespaces terminal:
export WANDB_API_KEY=your_api_key

# 5. Build and run
docker build -t mrpc-training .
docker run --rm -e WANDB_API_KEY=$WANDB_API_KEY mrpc-training
```



## Results

### Performance Metrics

| Metric | Value
|--------|-------
| **Validation Accuracy** | 85.29%
| **F1 Score** | 89.23%
| **Validation Loss** | 0.4089
| **Training Time (CPU)** | ~37 minutes | 3 epochs, batch size 32 |


### Code Structure

**`train.py` contains:**
- `GLUEDataModule`: Data loading, tokenization, and batching
- `GLUETransformer`: Model definition, training/validation logic
- `parse_args()`: Command-line argument parsing
- `main()`: Training orchestration and W&B integration



Made for MLOps Project 2 - Containerization
