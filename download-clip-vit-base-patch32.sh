#!/bin/sh
apt -y install -qq aria2
aria2c -c -x 16 -s 16 -k 1M https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/pytorch_model.bin \
-d ./checkpoints/clip-vit-base-patch32
mv ./checkpoints/clip-vit-base-patch32/a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f \
./checkpoints/clip-vit-base-patch32/pytorch_model.bin
