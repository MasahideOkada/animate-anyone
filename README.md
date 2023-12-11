# Animate Anyone
An attempt to implement [Animate Anyone](https://arxiv.org/abs/2311.17117) \
This repository is based on [diffusers 0.24.0](https://github.com/huggingface/diffusers/tree/v0.24.0) and [AnimateDiff](https://github.com/guoyww/AnimateDiff). \
work in progress

# train 1st stage
download checkpoints of `stable-diffusion-v1-5` and `clip-vit-base-patch32` and put them in corresponding folders in `checkpoints` directory, then 
run

```
accelerate launch --mixed_precision="fp16" train_1st_stage.py \
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
