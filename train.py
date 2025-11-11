#!/usr/bin/env python3
"""
Training script for MRPC paraphrase detection task
Converts the hyperparameter tuning notebook into a production-ready script
"""

import argparse
import os
from datetime import datetime
from typing import Optional

import datasets
import evaluate
import lightning as L
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)


class GLUEDataModule(L.LightningDataModule):
    """Data module for GLUE tasks"""
    
    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = "mrpc",
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str):
        self.dataset = datasets.load_dataset("glue", self.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        datasets.load_dataset("glue", self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def convert_to_features(self, example_batch, indices=None):
        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, padding="max_length", truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        return features


class GLUETransformer(L.LightningModule):
    """Lightning module for GLUE task fine-tuning"""
    
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        learning_rate: float = 9e-5,
        warmup_steps: int = 150,
        weight_decay: float = 0.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        scheduler_type: str = "cosine",
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.metric = evaluate.load(
            "glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )

        self.validation_step_outputs = []

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]
        self.validation_step_outputs.append({"loss": val_loss, "preds": preds, "labels": labels})
        return val_loss

    def on_validation_epoch_end(self):
        if self.hparams.task_name == "mnli":
            for i, output in enumerate(self.validation_step_outputs):
                # matched or mismatched
                split = self.hparams.eval_splits[i].split("_")[-1]
                preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x["loss"] for x in output]).mean()
                self.log(f"val_loss_{split}", loss, prog_bar=True)
                split_metrics = {
                    f"{k}_{split}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()
                }
                self.log_dict(split_metrics, prog_bar=True)
            self.validation_step_outputs.clear()
            return loss

        preds = torch.cat([x["preds"] for x in self.validation_step_outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in self.validation_step_outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, 
            lr=self.hparams.learning_rate, 
            betas=(self.hparams.beta1, self.hparams.beta2)
        )

        # Choose scheduler based on scheduler_type
        if self.hparams.scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        else:  # linear is default
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train a DistilBERT model on MRPC paraphrase detection task"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="distilbert-base-uncased",
        help="Path to pretrained model or model identifier from huggingface.co/models"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="mrpc",
        help="The name of the GLUE task to train on"
    )
    
    # Training arguments
    parser.add_argument(
        "--learning_rate",
        "--lr",
        type=float,
        default=9e-5,
        help="Learning rate (default: 8.5e-5, best from Optuna tuning)"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=150,
        help="Number of warmup steps (default: 100, best from Optuna tuning)"
    )
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="cosine",
        choices=["linear", "cosine"],
        help="Type of learning rate scheduler (default: linear, best from Optuna tuning)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for AdamW optimizer"
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Beta1 for AdamW optimizer"
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="Beta2 for AdamW optimizer"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="Maximum sequence length"
    )
    
    # System arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Accelerator type (auto, cpu, gpu, mps)"
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices to use"
    )
    
    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints"
    )
    
    # Experiment tracking arguments
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="mrpc-training",
        help="W&B project name"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name (default: auto-generated)"
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        nargs="+",
        default=["production", "best-params"],
        help="W&B tags for the run"
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default=None,
        help="W&B API key (can also use WANDB_API_KEY environment variable)"
    )
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Set random seed for reproducibility
    L.seed_everything(args.seed)
    
    # Initialize W&B only if not disabled
    wandb_run = None
    logger = None
    
    if not args.no_wandb:
        # Login to W&B if API key is provided
        if args.wandb_api_key:
            wandb.login(key=args.wandb_api_key)
        elif "WANDB_API_KEY" in os.environ:
            wandb.login(key=os.environ["WANDB_API_KEY"])
        else:
            print("Warning: No W&B API key found. Running without W&B logging.")
            print("Use --no_wandb flag to suppress this warning, or provide API key with:")
            print("  --wandb_api_key YOUR_KEY")
            print("  or export WANDB_API_KEY=YOUR_KEY")
            args.no_wandb = True  # Disable W&B if no key found
    
    if not args.no_wandb:
        # Initialize W&B run
        run_name = args.wandb_run_name or f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            tags=args.wandb_tags,
            config=vars(args),
        )
        logger = WandbLogger(experiment=wandb_run)
        print(f"W&B logging enabled - Run: {run_name}")
    else:
        print("W&B logging disabled")
    
    print(f"\nStarting training run")
    print(f"Hyperparameters: lr={args.learning_rate}, warmup={args.warmup_steps}, scheduler={args.scheduler_type}")
    
    # Setup data module
    print(f"\nLoading {args.task_name} dataset...")
    dm = GLUEDataModule(
        model_name_or_path=args.model_name_or_path,
        task_name=args.task_name,
        max_seq_length=args.max_seq_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )
    dm.setup("fit")
    print(f"Dataset loaded. Number of labels: {dm.num_labels}")
    
    # Initialize model
    print(f"\nInitializing model: {args.model_name_or_path}")
    model = GLUETransformer(
        model_name_or_path=args.model_name_or_path,
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        scheduler_type=args.scheduler_type,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Setup trainer
    print(f"\nInitializing trainer...")
    print(f"Accelerator: {args.accelerator}, Devices: {args.devices}")
    print(f"Epochs: {args.epochs}")
    
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=logger,
        default_root_dir=args.checkpoint_dir,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Train the model
    print(f"\nStarting training...")
    trainer.fit(model, datamodule=dm)
    
    # Get final metrics
    final_metrics = trainer.callback_metrics
    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"{'='*50}")
    print(f"Final validation loss: {final_metrics.get('val_loss', 'N/A')}")
    print(f"Final accuracy: {final_metrics.get('accuracy', 'N/A')}")
    print(f"Final F1 score: {final_metrics.get('f1', 'N/A')}")
    print(f"{'='*50}")
    
    # Finish W&B run if it was initialized
    if wandb_run is not None:
        wandb_run.finish()
        print(f"\nW&B run: {wandb_run.url if hasattr(wandb_run, 'url') else 'N/A'}")
    
    print(f"\nModel checkpoints saved to: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()