import os, csv
from typing import Dict, List,Optional, Tuple, Union

import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms.functional import crop

from decord import VideoReader

class VideoDataset(Dataset):
    def __init__(
        self,
        csv_path: Union[str, os.PathLike],
        video_dir: Union[str, os.PathLike],
        col_name: str = "name",
        pose_suffix: str = "pose",
        sample_size: int = 768,
        sample_stride: int = 4,
        sample_n_frames: int = 24,
        seed: Optional[int] = None,
    ):
        with open(csv_path, "r") as f:
            self.dataset = list(csv.DictReader(f))
        self.length = len(self.dataset)

        self.video_dir = video_dir
        self.col_name = col_name
        self.pose_suffix = pose_suffix
        self.sample_size = sample_size
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.only_single_frame = sample_n_frames == 1

        self.preproc = transforms.Compose(
            [
                transforms.Resize(
                    sample_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
                ),
                transforms.CenterCrop(sample_size),
            ]
        )

        self.rng = np.random.default_rng(seed)
    
    def __len__(self) -> int:
        return self.length
    
    def get_batch(self, idx: int) -> Tuple[torch.Tensor, ...]:
        video_dict = self.dataset[idx]
        video_name = video_dict[self.col_name]
        
        video_path = os.path.join(self.video_dir, f"{video_name}.mp4")
        pose_video_path = os.path.join(self.video_dir, f"{video_name}_{self.pose_suffix}.mp4")
        video_reader = VideoReader(video_path)
        pose_video_reader = VideoReader(pose_video_path)
        video_length = len(video_reader)

        if not self.only_single_frame:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx = self.rng.integers(video_length - clip_length + 1)
            batch_index = np.linspace(
                start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int
            )
        else:
            batch_index = [self.rng.integers(video_length)]
        
        ref_index = [self.rng.integers(video_length)]

        pixel_values = torch.from_numpy(
            video_reader.get_batch(batch_index).asnumpy()
        ).permute(0, 3, 1, 2)
        pose_pixel_values = torch.from_numpy(
            pose_video_reader.get_batch(batch_index).asnumpy()
        ).permute(0, 3, 1, 2)
        ref_pixel_values = torch.from_numpy(
            video_reader.get_batch(ref_index).asnumpy()
        ).permute(0, 3, 1, 2)
        
        del video_reader

        return pixel_values, pose_pixel_values, ref_pixel_values

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        while True:
            try:
                pixel_values, pose_pixel_values, ref_pixel_values = self.get_batch(idx)
                break
            except Exception as e:
                idx = self.rng.integers(self.length)

        pixel_values = self.preproc(pixel_values) / 255.0
        pose_pixel_values = self.preproc(pose_pixel_values) / 255.0
        ref_pixel_values = self.preproc(ref_pixel_values) / 255.0

        pixel_values = 2.0 * pixel_values - 1.0
        pose_pixel_values = 2.0 * pose_pixel_values - 1.0

        return {
            "pixel_values": pixel_values,
            "pose_pixel_values": pose_pixel_values,
            "ref_pixel_values": ref_pixel_values,
        }

def collate_fn(examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    `pixel_values`: normalized [-1, 1] \\
    `pose_pixel_values`: normalized [-1, 1] \\
    `ref_pixel_values`: unnormalized [0, 1]
    """
    pixel_values = torch.stack([example["pixel_values"] for example in examples]).flatten(0, 1)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    pose_pixel_values = torch.stack([example["pose_pixel_values"] for example in examples]).flatten(0, 1)
    pose_pixel_values = pose_pixel_values.to(memory_format=torch.contiguous_format).float()

    ref_pixel_values = torch.stack([example["ref_pixel_values"] for example in examples]).flatten(0, 1)
    ref_pixel_values = ref_pixel_values.to(memory_format=torch.contiguous_format).float()

    return {
        "pixel_values": pixel_values,
        "pose_pixel_values": pose_pixel_values,
        "ref_pixel_values": ref_pixel_values,
    }
