from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from src.training.data_module import PackedDataModule
from src.training.generation import generate_text
from src.training.lightning_module import GPTLightningModule
from src.tokenization.bpe import BpeTokenizer


def init_clearml(config) -> None:
    if not bool(config.logging.clearml_enabled):
        return
    try:
        from clearml import Task
    except ImportError as exc:
        raise RuntimeError("ClearML is enabled, but package clearml is not installed.") from exc
    task = Task.init(
        project_name=str(config.logging.clearml_project),
        task_name=str(config.logging.experiment_name),
    )
    task.connect(OmegaConf.to_container(config, resolve=True))


def cmd_train(args: argparse.Namespace) -> None:
    config = OmegaConf.load(args.config)
    if args.data:
        config.paths.train_data = args.data
    if args.resume_from_checkpoint:
        config.paths.resume_from_checkpoint = args.resume_from_checkpoint

    L.seed_everything(int(config.seed), workers=True)
    init_clearml(config)
    tokenizer = BpeTokenizer.load(config.paths.tokenizer)
    if tokenizer.vocab_size != int(config.model.vocab_size):
        raise ValueError(
            "Tokenizer vocabulary and model config do not match: "
            f"{tokenizer.vocab_size} != {config.model.vocab_size}. "
            "Re-run lab1 tokenize and pack with the intended BPE vocabulary size."
        )

    data_module = PackedDataModule(
        data_path=config.paths.train_data,
        batch_size=int(config.training.batch_size),
        val_fraction=float(config.training.val_fraction),
        num_workers=int(config.training.num_workers),
        seed=int(config.seed),
    )
    model = GPTLightningModule(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(config.paths.checkpoint_dir),
        filename="final-{epoch:02d}-{val_perplexity:.2f}",
        monitor="val_perplexity",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    logger = TensorBoardLogger(
        save_dir=str(config.paths.log_dir),
        name=str(config.logging.tensorboard_name),
    )
    trainer = L.Trainer(
        max_epochs=int(config.training.max_epochs),
        accelerator=str(config.training.accelerator),
        devices=config.training.devices,
        precision=config.training.precision,
        logger=logger,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="step")],
        gradient_clip_val=float(config.training.gradient_clip_val),
        log_every_n_steps=int(config.training.log_every_n_steps),
        accumulate_grad_batches=int(config.training.accumulate_grad_batches),
        fast_dev_run=bool(args.fast_dev_run),
    )
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=args.resume_from_checkpoint,
    )
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")


def cmd_generate(args: argparse.Namespace) -> None:
    config = OmegaConf.load(args.config)
    text = generate_text(
        checkpoint_path=args.checkpoint,
        config=config,
        prompt=args.prompt,
        tokenizer_path=args.tokenizer or str(config.paths.tokenizer),
        max_new_tokens=args.max_new_tokens or int(config.generation.max_new_tokens),
        temperature=args.temperature or float(config.generation.temperature),
        top_k=args.top_k if args.top_k is not None else int(config.generation.top_k),
    )
    print(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lab 2 GPT-like model training CLI")
    parser.add_argument("--config", default="configs/lab2_gpt.yaml")
    subparsers = parser.add_subparsers(dest="command")

    train = subparsers.add_parser("train")
    train.add_argument("--config", default="configs/lab2_gpt.yaml")
    train.add_argument("--data")
    train.add_argument("--resume-from-checkpoint")
    train.add_argument("--fast-dev-run", action="store_true")
    train.set_defaults(func=cmd_train)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--config", default="configs/lab2_gpt.yaml")
    generate.add_argument("--checkpoint", required=True)
    generate.add_argument("--prompt", default="The model")
    generate.add_argument("--tokenizer")
    generate.add_argument("--max-new-tokens", type=int)
    generate.add_argument("--temperature", type=float)
    generate.add_argument("--top-k", type=int)
    generate.set_defaults(func=cmd_generate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
