#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.utils.checkpoint
import torchvision.transforms as transforms
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModel

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from animate_anyone.models.unet_2d_condition import UNet2DConditionModel
from animate_anyone.utils.reference_utils import save_reference_features


logger = get_logger(__name__)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="script to save ReferenceNet's hidden feature of reference images")
    parser.add_argument(
        "--pretrained_dir",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained checkpoint directory",
    )
    parser.add_argument(
        "--sd_dir",
        type=str,
        default="stable-diffusion-v1-5",
        required=False,
        help="sub folder of pretrained stable diffusion in the `pretrained_checkpoint_dir`",
    )
    parser.add_argument(
        "--clip_dir",
        type=str,
        default="clip-vit-base-patch32",
        required=False,
        help="sub folder of pretrained CLIP encoder in the `pretrained_checkpoint_dir`",
    )
    parser.add_argument(
        "--refnet_dir",
        type=str,
        default=None,
        required=True,
        help="a folder of pretrained ReferenceNet",
    )
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        default=None,
        required=True,
        help="a folder of pretrained ReferenceNet",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ref-features",
        help="The output directory where the reference net hidden features will be written.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    return args


def main(args):
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id
    
    clip_dir = os.path.join(args.pretrained_dir, args.clip_dir)
    # Load image encoder
    image_encoder = CLIPVisionModel.from_pretrained(clip_dir)
    clip_processor = CLIPImageProcessor.from_pretrained(clip_dir)
    
    # Load scheduler and vae
    sd_dir = os.path.join(args.pretrained_dir, args.sd_dir)
    vae = AutoencoderKL.from_pretrained(
        sd_dir, subfolder="vae", revision=args.revision, variant=args.variant
    )

    # Load ReferenceNet
    logger.info(f"Loading ReferenceNet weights from {args.refnet_dir}")
    refnet = UNet2DConditionModel.from_pretrained(
        args.refnet_dir, subfolder="referencenet", use_safetensors=True
    )

    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    refnet.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            refnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    
    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(refnet).dtype != torch.float32:
        raise ValueError(
            f"Unet loaded as datatype {accelerator.unwrap_model(refnet).dtype}. {low_precision_error_string}"
        )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, image_encoder, referencenet and pose_guider to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    refnet.to(accelerator.device, dtype=weight_dtype)

    ref_images = args.images

    def get_filename(path) -> str:
        basename = os.path.basename(path)
        name, _ = os.path.splitext(basename)
        return name

    logger.info("***** Starting *****")
    logger.info(f"  Num images = {len(ref_images)}")

    progress_bar = tqdm(
        range(0, len(ref_images)),
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for img_path in ref_images:
        image = Image.open(img_path).convert("RGB")
        ref_pixel_values = transforms.functional.to_tensor(image)[None, :]

        # Get CLIP embedding of reference image for cross attention
        ref_pixel_values = ref_pixel_values.to(accelerator.device, dtype=weight_dtype)
        ref_clip_values = clip_processor(
            ref_pixel_values, do_rescale=False, return_tensors="pt"
        )["pixel_values"].to(accelerator.device, dtype=weight_dtype)
        encoder_hidden_states = image_encoder(pixel_values=ref_clip_values).last_hidden_state

        # Get RefereceNet outputs
        ref_pixel_values = 2.0 * ref_pixel_values - 1.0
        ref_latents = vae.encode(ref_pixel_values).latent_dist.sample()
        ref_latents = ref_latents * vae.config.scaling_factor
        reference_hidden_states = refnet(
            ref_latents, 0, encoder_hidden_states, return_dict=False
        )[-1]

        # Save
        image_name = get_filename(img_path)
        sf_name = f"{image_name}_ref.safetensors"
        output_path = sf_name if args.output_dir is None else os.path.join(args.output_dir, sf_name)
        save_reference_features(reference_hidden_states, output_path)
        logger.info(f"Saved the reference features of {img_path} to {output_path}")
        progress_bar.update(1)


if __name__ == "__main__":
    args = parse_args()
    main(args)
