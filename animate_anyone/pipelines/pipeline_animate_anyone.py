# Copyright 2023 The HuggingFace Team. All rights reserved.
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
# limitations under the License.

import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from transformers import CLIPImageProcessor, CLIPVisionModel

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.schedulers import DDPMScheduler
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from ..models.pose_guider import PoseGuider
from ..models.unet_2d_condition import UNet2DConditionModel
from ..models.unet_3d_condition import UNet3DConditionModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Adapted from https://github.com/huggingface/diffusers/blob/76c645d3a641c879384afcb43496f0b7db8cc5cb/src/diffusers/pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py#L35
def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]

# Adapted from https://github.com/huggingface/diffusers/blob/76c645d3a641c879384afcb43496f0b7db8cc5cb/src/diffusers/pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py#L43
def tensor2vid(video: torch.Tensor, processor, output_type="np"):
    # Based on:
    # https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/pipelines/multi_modal/text_to_video_synthesis_pipeline.py#L78

    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    return outputs


@dataclass
class AnimateAnyonePipelineOutput(BaseOutput):
    r"""
    Output class for Animate Anyone pipeline.

    Args:
        frames (`[List[PIL.Image.Image]`, `np.ndarray`]):
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
    """

    frames: Union[List[PIL.Image.Image], np.ndarray]


class AnimateAnyonePipeline(DiffusionPipeline):
    r"""
    Pipeline to generate video from an input image using Animate Anyone.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        image_encoder ([`~transformers.CLIPVisionModel`]):
            Frozen CLIP image-encoder ([laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)).
        unet ([`UNet3DConditionModel`]):
            A `UNet3DConditionModel` to denoise the encoded image latents.
        referencenet ([`UNet2DConditionModel`])
            A `UNet2DConditionModel` to encode reference images.
        pose_guider ([`PoseGuider`])
            A `PoseGuider` to encode pose guidances
        scheduler ([`DDPMScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        clip_processor ([`~transformers.CLIPImageProcessor,`]):
            A `CLIPImageProcessor,` to extract features from generated images.
    """

    model_cpu_offload_seq = "image_encoder->pose_guider->referencenet->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        vae: AutoencoderKL,
        image_encoder: CLIPVisionModel,
        unet: UNet3DConditionModel,
        referencenet: UNet2DConditionModel,
        pose_guider: PoseGuider,
        scheduler: DDPMScheduler,
        clip_processor: CLIPImageProcessor,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            referencenet=referencenet,
            pose_guider=pose_guider,
            scheduler=scheduler,
            clip_processor=clip_processor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)


    def _encode_image(self, image, device, num_videos_per_prompt, do_classifier_free_guidance):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.image_processor.pil_to_numpy(image)
            image = self.image_processor.numpy_to_pt(image)

        image = image.to(device=device, dtype=dtype)
        image_proc = self.clip_processor(image, do_rescale=False, return_tensors="pt")
        image_embeddings = self.image_encoder(**image_proc).last_hidden_state
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, _, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        return image_embeddings
    
    def _encode_pose_sequence(self, pose_sequence, batch_size, device, num_videos_per_prompt, do_classifier_free_guidance):
        do_normalise = True
        if isinstance(pose_sequence[0], list):
            pose_sequences = []
            for ps in pose_sequence:
                ps = self.image_processor.pil_to_numpy(ps)
                ps = self.image_processor.numpy_to_pt(ps)
                pose_sequences.append(ps)
            pose_sequence = torch.stack(pose_sequences)
        elif isinstance(pose_sequence, list):
            pose_sequence = self.image_processor.pil_to_numpy(pose_sequence)
            pose_sequence = self.image_processor.numpy_to_pt(pose_sequence)
            pose_sequence = pose_sequence.unsqueeze(0)
            pose_sequence = pose_sequence.repeat(batch_size, 1, 1, 1, 1)
        else:
            do_normalise = False
        
        pose_sequence = pose_sequence.to(device)
        batch_size, num_frames, channel, height, width = pose_sequence.shape
        pose_sequence = pose_sequence.reshape(batch_size*num_frames, channel, height, width)
        pose_embeddings = self.pose_guider(pose_sequence, do_normalize=do_normalise)
        pose_embeddings = pose_embeddings.repeat(num_videos_per_prompt, 1, 1, 1)
        
        if do_classifier_free_guidance:
            negative_pose_embeddings = torch.zeros_like(pose_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            pose_embeddings = torch.cat([negative_pose_embeddings, pose_embeddings])
        
        return pose_embeddings

    # Adapted from https://github.com/huggingface/diffusers/blob/76c645d3a641c879384afcb43496f0b7db8cc5cb/src/diffusers/pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py#L157
    def _encode_vae_image(
        self,
        image: torch.Tensor,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
    ):
        image = image.to(device=device)
        image_latents = self.vae.encode(image).latent_dist.mode()

        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_latents = torch.cat([negative_image_latents, image_latents])

        # duplicate image_latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)

        return image_latents

    # Adapted from https://github.com/huggingface/diffusers/blob/76c645d3a641c879384afcb43496f0b7db8cc5cb/src/diffusers/pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py#L208
    def decode_latents(self, latents, num_frames, decode_chunk_size=14):
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        #latents = latents.flatten(0, 1)

        latents = 1 / self.vae.config.scaling_factor * latents

        accepts_num_frames = "num_frames" in set(inspect.signature(self.vae.forward).parameters.keys())

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in

            frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)

        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames

    
    def check_inputs(self, image, poses, height, width):
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )
        
        if not isinstance(poses, torch.Tensor) and not isinstance(poses, list):
            raise ValueError(
                "`poses` has to be of type `torch.FloatTensor` or `List[PIL.Image.Image]` or `List[List[PIL.Image.Image]]` but is"
                f" {type(image)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    # Modified from https://github.com/huggingface/diffusers/blob/76c645d3a641c879384afcb43496f0b7db8cc5cb/src/diffusers/pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py#L250
    def prepare_latents(
        self,
        batch_size,
        num_frames,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_frames,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        latents = latents.flatten(0, 1)
        return latents

    # Adapted from https://github.com/huggingface/diffusers/blob/76c645d3a641c879384afcb43496f0b7db8cc5cb/src/diffusers/pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py#L284
    @property
    def guidance_scale(self):
        return self._guidance_scale

    # Adapted from https://github.com/huggingface/diffusers/blob/76c645d3a641c879384afcb43496f0b7db8cc5cb/src/diffusers/pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py#L292
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    # Adapted from https://github.com/huggingface/diffusers/blob/76c645d3a641c879384afcb43496f0b7db8cc5cb/src/diffusers/pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py#L296
    @property
    def num_timesteps(self):
        return self._num_timesteps
    
    def get_num_frames(self, pose_sequence):
        if isinstance(pose_sequence, torch.FloatTensor):
            num_frames = pose_sequence.shape[1]
        elif isinstance(pose_sequence[0], list):
            num_frames = len(pose_sequence[0])
            for ps in pose_sequence:
                num_frames_each = len(ps)
                if num_frames != num_frames_each:
                    raise ValueError("number of frames must be the same in the batch")
        else:
            num_frames = len(pose_sequence)
        
        return num_frames

    @torch.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        pose_sequence: Union[List[PIL.Image.Image], List[List[PIL.Image.Image]], torch.FloatTensor],
        height: int = 576,
        width: int = 1024,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image or images for reference. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
            pose_sequence (`List[PIL.Image.Image]` or `List[List[PIL.Image.Image]]` or `torch.FloatTensor`):
                pose sequences to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
            fps (`int`, *optional*, defaults to 7):
                Frames per second. The rate at which the generated images shall be exported to a video after generation.
                Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                The motion bucket ID. Used as conditioning for the generation. The higher the number the more motion will be in the video.
            noise_aug_strength (`int`, *optional*, defaults to 0.02):
                The amount of noise added to the init image, the higher it is the less the video will look like the init image. Increase it for more motion.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. The higher the chunk size, the higher the temporal consistency
                between frames, but also the higher the memory consumption. By default, the decoder will decode all frames at once
                for maximal quality. Reduce `decode_chunk_size` to reduce memory usage.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`AnimateAnyonePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list of list with the generated frames.

        Examples:
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, pose_sequence, height, width)

        num_frames = self.get_num_frames(pose_sequence)
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames        

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
            if isinstance(pose_sequence[0], list):
                pose_batch_size = len(pose_sequence)
                if batch_size != pose_batch_size:
                    raise ValueError("batch sizes must be the same for the reference images and pose sequences")
        else:
            batch_size = image.shape[0]
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = max_guidance_scale > 1.0

        # 3. Encode reference image using VAE as inputs to ReferenceNet
        image = self.image_processor.preprocess(image, height=height, width=width)

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        image_latents = self._encode_vae_image(
            image, device, num_videos_per_prompt, do_classifier_free_guidance
        )
        image_latents = image_latents.to(image_embeddings.dtype)

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # 4. Encode reference image for cross attention
        image_embeddings = self._encode_image(
            image, device, num_videos_per_prompt, do_classifier_free_guidance
        )

        # 5. Get ReferenceNet hidden states
        reference_hidden_states = self.referencenet(
            image_latents, 0, image_embeddings, return_dict=False
        )[-1]

        # 6. Encode pose sequences
        pose_embeddings = self._encode_pose_sequence(
            pose_sequence, batch_size, device, num_videos_per_prompt, do_classifier_free_guidance
        )

        # 7. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 8. Prepare latent variables
        num_channels_latents = pose_embeddings.shape[1]
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 9. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale

        # 10. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # fuse latents and pose embeddings
                latents = latents + pose_embeddings
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=image_embeddings,
                    num_frames=num_frames,
                    reference_hidden_states=reference_hidden_states,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return AnimateAnyonePipelineOutput(frames=frames)
