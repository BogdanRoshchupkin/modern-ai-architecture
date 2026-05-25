from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split


class PackedJsonlDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, path: str | Path) -> None:
        input_ids: list[list[int]] = []
        segment_ids: list[list[int]] = []
        with Path(path).open("r", encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    row = json.loads(line)
                    input_ids.append(row["input_ids"])
                    segment_ids.append(row["attention_mask"])
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.segment_ids = torch.tensor(segment_ids, dtype=torch.long)

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "segment_ids": self.segment_ids[index],
        }


class PackedDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str | Path,
        batch_size: int,
        val_fraction: float = 0.1,
        num_workers: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.num_workers = num_workers
        self.seed = seed
        self.train_dataset: Dataset[dict[str, torch.Tensor]] | None = None
        self.val_dataset: Dataset[dict[str, torch.Tensor]] | None = None

    def setup(self, stage: str | None = None) -> None:
        dataset = PackedJsonlDataset(self.data_path)
        val_size = max(1, int(len(dataset) * self.val_fraction))
        train_size = len(dataset) - val_size
        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )
