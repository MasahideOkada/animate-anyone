#!/bin/sh
apt -y install -qq aria2
aria2c -c -x 16 -s 16 -k 1M https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt \
-d ./checkpoints/motion-modules/
mv ./checkpoints/motion-modules/69ed0f5fef82b110aca51bcab73b21104242bc65d6ab4b8b2a2a94d31cad1bf0 \
./checkpoints/motion-modules/mm_sd_v15_v2.ckpt
