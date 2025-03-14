from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

torch.set_num_threads(1)


@dataclass
class DataConfig:
    filelist_path: str
    sampling_rate: int
    num_samples: int
    batch_size: int
    num_workers: int


class VocosDataModule(LightningDataModule):
    def __init__(self, train_params: DataConfig, val_params: DataConfig):
        super().__init__()
        self.train_config = train_params
        self.val_config = val_params

    def _get_dataloder(self, cfg: DataConfig, train: bool):
        dataset = VocosDataset(cfg, train=train)
        dataloader = DataLoader(
            dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=train, pin_memory=True,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.train_config, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.val_config, train=False)


class CosyvoiceDataModule(LightningDataModule):
    def __init__(self, train_params: DataConfig, val_params: DataConfig):
        super().__init__()
        self.train_config = train_params
        self.val_config = val_params

    def _get_dataloder(self, cfg: DataConfig, train: bool):
        dataset = CosyvoiceDataset(cfg, train=train)
        dataloader = DataLoader(
            dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=train, pin_memory=True,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.train_config, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.val_config, train=False)


class CosyvoiceDataset(Dataset):
    def __init__(self, cfg: DataConfig, train: bool):
        self.filelist = load_dataset('json', data_files=cfg.filelist_path, split='train')
        self.sampling_rate = cfg.sampling_rate
        self.num_samples = cfg.num_samples
        self.train = train

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, index: int) -> torch.Tensor:
        sample = self.filelist[index]
        # speech_token: (1,-1)
        # embedding: (1,192)
        # speech_feat: (1,-1)
        # audio: str
        y, sr = torchaudio.load(sample['audio'])
        speech_token = np.array(sample['speech_token'])
        if self.train:
            start_idx = np.random.randint(low=0, high=int(y.size(-1) / self.sampling_rate * 25) - int(
                self.num_samples / self.sampling_rate * 25) + 1)
            end_idx = start_idx + int(self.num_samples / self.sampling_rate * 25)
            speech_token = speech_token[:, start_idx:end_idx]
            y = y[:,
                int(start_idx / 25 * self.sampling_rate): int(start_idx / 25 * self.sampling_rate) + self.num_samples]
        else:
            start_idx = 0
            end_idx = start_idx + int(self.num_samples / self.sampling_rate * 25)
            speech_token = speech_token[:, start_idx:end_idx]
            y = y[:, :self.num_samples]

        speech_token = torch.tensor(speech_token[0], dtype=torch.int32)
        if 'embedding' in sample:
            speaker_embedding = torch.tensor(sample['embedding'][0], dtype=torch.float32)
        else:
            speaker_embedding = torch.zeros((1, 192), dtype=torch.float32)

        return y[0], speech_token, speaker_embedding


class VocosDataset(Dataset):
    def __init__(self, cfg: DataConfig, train: bool):
        with open(cfg.filelist_path, encoding='utf8') as f:
            self.filelist = f.read().splitlines()
        self.sampling_rate = cfg.sampling_rate
        self.num_samples = cfg.num_samples
        self.train = train

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, index: int) -> torch.Tensor:
        audio_path = self.filelist[index]
        y, sr = torchaudio.load(audio_path)
        if y.size(0) > 1:
            # mix to mono
            y = y.mean(dim=0, keepdim=True)
        gain = np.random.uniform(-1, -6) if self.train else -3
        y, _ = torchaudio.sox_effects.apply_effects_tensor(y, sr, [["norm", f"{gain:.2f}"]])
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
        if y.size(-1) < self.num_samples:
            pad_length = self.num_samples - y.size(-1)
            padding_tensor = y.repeat(1, 1 + pad_length // y.size(-1))
            y = torch.cat((y, padding_tensor[:, :pad_length]), dim=1)
        elif self.train:
            start = np.random.randint(low=0, high=y.size(-1) - self.num_samples + 1)
            y = y[:, start: start + self.num_samples]
        else:
            # During validation, take always the first segment for determinism
            y = y[:, : self.num_samples]

        return y[0]
