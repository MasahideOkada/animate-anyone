#!/bin/sh
apt -y install -qq aria2
aria2c -c -x 16 -s 16 -k 1M https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/text_encoder/model.fp16.safetensors \
-d ./checkpoints/stable-diffusion-v1-5/text_encoder/bin
aria2c -c -x 16 -s 16 -k 1M https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors \
-d ./checkpoints/stable-diffusion-v1-5/unet/bin
aria2c -c -x 16 -s 16 -k 1M https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors \
-d ./checkpoints/stable-diffusion-v1-5/vae/bin
mv ./checkpoints/stable-diffusion-v1-5/text_encoder/bin/77795e2023adcf39bc29a884661950380bd093cf0750a966d473d1718dc9ef4e \
./checkpoints/stable-diffusion-v1-5/text_encoder/model.safetensors
mv ./checkpoints/stable-diffusion-v1-5/unet/bin/c83908253f9a64d08c25fc90874c9c8aef9a329ce1ca5fb909d73b0c83d1ea21 \
./checkpoints/stable-diffusion-v1-5/unet/diffusion_pytorch_model.safetensors
mv ./checkpoints/stable-diffusion-v1-5/vae/bin/4fbcf0ebe55a0984f5a5e00d8c4521d52359af7229bb4d81890039d2aa16dd7c \
./checkpoints/stable-diffusion-v1-5/vae/diffusion_pytorch_model.safetensors
