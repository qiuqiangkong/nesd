# NeSD: Neural Sound Field Decomposition

This repository provides a PyTorch implementation of Neural Sound Field Decomposition (NeSD).
The system takes microphone array recordings as input and produces a 360° × 160° panoramic map representing the probability distribution of active sound events in the environment. NeSD is designed to handle arbitrary microphone array configurations, diverse acoustic environments, and an unknown number of simultaneous sound sources. NeSD can be applied to a variety of spatial audio tasks, including sound source localization, distance estimation, and spatial source separation.

<img width="496" height="280" alt="screenshot-20251106-103938" src="https://github.com/user-attachments/assets/8cd4081d-1cb1-4d02-84d4-c632b6daf030" />



## 0. Install dependencies

```bash
# Clone the repo
git clone https://github.com/qiuqiangkong/nesd
cd nesd

# Install Python environment
conda create --name nesd python=3.10

# Activate environment
conda activate nesd

# Install Python packages dependencies
bash env.sh
```

## 1. Prepare audio datasets

Download VCTK speech dataset from https://datashare.ed.ac.uk/handle/10283/3443. The downloaded dataset looks like:

<pre>
vctk
├── wav48 (109 speakers)
│   ├── p225 (231 files)
│   │   └── ...
│   ├── p229 (379 files)
│   │   └── ...
│   └── ...
...
</pre>

Prepare audio 2s audio segments for training.

```bash
# Prepare 2s segments
bash ./scripts/prepare_audios/vctk.sh
```

## 2. Prepare room environments

```python
python -m rooms.shoebox_ism \
  --mic_csv="./assets/mics/em32.csv" \
  --data_num=100 \
  --out_dir="./results/rooms/ism"

This step can be skipped when using online training.

```

## 3. Prepare microphones

```python
python -m mics.eigenmike \
  --sphere_type="rigid" \
  --out_path="./results/mics/eigenmike.h5"
```

## 4. Train

```python
CUDA_VISIBLE_DEVICES=0 python train.py --config="./configs/convdnn.yaml"
```

## 5. Inference

Download test audio:

```bash
bash ./scripts/download_test_audios/em32.sh
```

```python
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --config="./configs/convdnn.yaml" \
  --ckpt_path="./checkpoints/train/convdnn/step=90000.pth" \
  --audio_path="./test_audios/fold4_room8_mix003.wav" \
  --video_path="./test_audios/fold4_room8_mix003.mp4" \
  --out_path="./out.mp4"
```

## Results

After training on one RTX4090 GPU card for 12 hours, the results look like:

https://github.com/user-attachments/assets/66d18f79-0fb1-4d7a-a3c5-154c5497d567

