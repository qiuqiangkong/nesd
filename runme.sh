
WORKSPACE="${HOME}/workspaces/nesd"

#
VCTK_DATASET_DIR="/datasets/vctk"
SAMPLE_RATE=24000
SEGMENT_SECONDS=2.0
python nesd/dataset_creation/vctk.py \
	--dataset_dir=$VCTK_DATASET_DIR \
	--split="train" \
	--sample_rate=$SAMPLE_RATE \
	--segment_seconds=$SEGMENT_SECONDS \
	--output_dir="${WORKSPACE}/audios/vctk_2s_segments"

#
TAU_NOISE_AUDIOS_DIR="/datasets/tau-srir/TAU-SNoise_DB"
CLIP_SECONDS=60.

python nesd/dataset_creation/tau_noise.py \
	--audios_dir=$TAU_NOISE_AUDIOS_DIR \
	--split="train" \
	--sample_rate=$SAMPLE_RATE \
	--clip_seconds=$CLIP_SECONDS \
	--output_dir="${WORKSPACE}/audios/tau_noise"


python nesd/mic_prep/rigid_sphere.py \
	--mic_spatial_irs_path="${WORKSPACE}/mic_spatial_irs/rigid_sphere.pkl"
	

python nesd/train.py \
	--workspace=$WORKSPACE \
	--config_yaml="./scripts/configs/01a.yaml"


python nesd/inference.py inference \
	--workspace=$WORKSPACE \
	--config_yaml="./scripts/configs/01a.yaml" \
	--checkpoint_path="/home/qiuqiangkong/workspaces/nesd/checkpoints/NeSD/step=100000.pth"
	
### --- evaluate dcase2019 ---
CUDA_VISIBLE_DEVICES=6 python evaluate/dcase2019_task3.py inference \
	--workspace=$WORKSPACE \
	--config_yaml="./scripts/configs/31a.yaml" \
	--checkpoint_path="/home/qiuqiangkong/workspaces/nesd/checkpoints/train/31a/step=300000.pth"

# python evaluate/dcase2019_task3.py write_loc_csv
# python evaluate/dcase2019_task3.py write_loc_csv_with_sed

# optional
python evaluate/dcase2019_task3.py plot_panaroma

# Write segments and locs to classify
python evaluate/dcase2019_task3.py panaroma_to_events

CUDA_VISIBLE_DEVICES=3 python sed/inference_d19t3.py inference --model_name=CRnn2

CUDA_VISIBLE_DEVICES=0 python evaluate/dcase2019_task3.py segs_sep \
	--workspace=$WORKSPACE \
	--config_yaml="./scripts/configs/43b.yaml" \
	--checkpoint_path="/home/qiuqiangkong/workspaces/nesd/checkpoints/train/43b/step=900000.pth"

CUDA_VISIBLE_DEVICES=0 python evaluate/dcase2019_task3.py segs_distance \
	--workspace=$WORKSPACE \
	--config_yaml="./scripts/configs/43b.yaml" \
	--checkpoint_path="/home/qiuqiangkong/workspaces/nesd/checkpoints/train/43b/step=900000.pth"

CUDA_VISIBLE_DEVICES=3 python sed/inference_d19t3.py inference_many --model_name=CRnn2

python evaluate/dcase2019_task3.py combine_results

python evaluate/d19t3_test.py

###
CUDA_VISIBLE_DEVICES=6 python evaluate/dcase2019_task3.py inference_distance \
	--workspace=$WORKSPACE \
	--config_yaml="./scripts/configs/31a.yaml" \
	--checkpoint_path="/home/qiuqiangkong/workspaces/nesd/checkpoints/train/31a/step=300000.pth"

CUDA_VISIBLE_DEVICES=6 python evaluate/dcase2019_task3.py inference_sep \
	--workspace=$WORKSPACE \
	--config_yaml="./scripts/configs/31a.yaml" \
	--checkpoint_path="/home/qiuqiangkong/workspaces/nesd/checkpoints/train/31a/step=300000.pth"

# evaluate dcase2020, no distance in the labels.
CUDA_VISIBLE_DEVICES=6 python evaluate/dcase2020_task3.py inference \
	--workspace=$WORKSPACE \
	--config_yaml="./scripts/configs/31a.yaml" \
	--checkpoint_path="/home/qiuqiangkong/workspaces/nesd/checkpoints/train/31a/step=300000.pth"
	
python evaluate/dcase2020_task3.py plot_panaroma \
	--workspace=$WORKSPACE

CUDA_VISIBLE_DEVICES=6 python evaluate/dcase2020_task3.py inference_depth \
	--workspace=$WORKSPACE \
	--config_yaml="./scripts/configs/31a.yaml" \
	--checkpoint_path="/home/qiuqiangkong/workspaces/nesd/checkpoints/train/31a/step=300000.pth"

CUDA_VISIBLE_DEVICES=6 python evaluate/dcase2020_task3.py inference_sep \
	--workspace=$WORKSPACE \
	--config_yaml="./scripts/configs/31a.yaml" \
	--checkpoint_path="/home/qiuqiangkong/workspaces/nesd/checkpoints/train/31a/step=300000.pth"


####
python nesd/train_old.py \
	--workspace=$WORKSPACE \
	--config_yaml="./scripts/configs/01a.yaml"


##### SED
CUDA_VISIBLE_DEVICES=2 python sed/train_d19t3.py


