
WORKSPACE="${HOME}/workspaces/nesd"

VCTK_DATASET_DIR="/datasets/vctk"
SAMPLE_RATE=24000
SEGMENT_SECONDS=2.0
python nesd/dataset_creation/vctk.py \
	--dataset_dir=$VCTK_DATASET_DIR \
	--split="train" \
	--sample_rate=$SAMPLE_RATE \
	--segment_seconds=$SEGMENT_SECONDS \
	--output_dir="${WORKSPACE}/audios/vctk_2s_segments"


python nesd/mic_prep/rigid_sphere.py \
	--mic_spatial_irs_path="${WORKSPACE}/mic_spatial_irs/rigid_sphere.pkl"
	

python nesd/train.py \
	--workspace=$WORKSPACE \
	--config_yaml="./scripts/configs/01a.yaml"
