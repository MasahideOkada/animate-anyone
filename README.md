# Animate Anyone
An attempt to implement [Animate Anyone](https://arxiv.org/abs/2311.17117) \
This repository is based on [diffusers 0.24.0](https://github.com/huggingface/diffusers/tree/v0.24.0) and [AnimateDiff](https://github.com/guoyww/AnimateDiff). \
work in progress

# train 1st stage
download checkpoints of [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) and [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) and put them in corresponding folders in `checkpoints` directory, then 
run

```
$ accelerate launch --mixed_precision="fp16" train_1st_stage.py \
--pretrained_dir "checkpoints" \
--train_data_dir "data" \
--csv_path "data.csv" \
--train_batch_size 64 \
--max_train_steps 30000 \
--output_dir "animany-stage1"
```

In the example above, `train_data_dir` is supposed to be structured like 

```
data/
├── video1.mp4
├── video1_pose.mp4
├── video2.mp4
├── video2_pose.mp4
├── video3.mp4
├── video3_pose.mp4
└── ...
```

and `csv_path` is the path to a csv file with `name` column containing non-pose video names to be used for training data, that is to say,

```
name
video1
video2
video3
...
```

# train 2nd stage

download the checkpoint of AnimateDiff's [mm_sd_v15_v2.ckpt](https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v15_v2.ckpt) and put it in corresponding folders in `checkpoints` directory, then 
run

```
$ accelerate launch --mixed_precision="fp16" train_2nd_stage.py \
--pretrained_dir "checkpoints" \
--stage1_dir "animany-stage1" \
--train_data_dir "data" \
--csv_path "data.csv" \
--train_batch_size 4 \
--num_frames 24 \
--max_train_steps 10000 \
--output_dir "animany-stage2"
```

# Save ReferenceNet features of image
Since ReferenceNet is needed only to extract features of a reference image, it will make sense to save those data in advance before the denoising process at inference time. If you want to do it, run

```
$ python save_reference_features.py \
--pretrained_dir "checkpoints" \
--refnet_dir "animany-stage1" \
--images "reference_image.png" \
--mixed_precision="fp16" \
--output_dir "ref-features"
```

This saves the reference feature data of input images in safetensors format.
