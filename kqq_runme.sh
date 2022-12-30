
WORKSPACE="/home/tiger/workspaces/nesd2"

SAMPLE_RATE=24000

# Paths
for SPLIT in "train" "test"
do  
    python3 ./nesd/dataset_creation/pack_audios_to_hdf5s/vctk.py \
        --dataset_dir="/home/tiger/datasets/vctk" \
        --split=$SPLIT \
        --hdf5s_dir="${WORKSPACE}/hdf5s/vctk/sr=${SAMPLE_RATE}/${SPLIT}" \
        --sample_rate=$SAMPLE_RATE \
        --segment_seconds=3.0
done

# Train
CUDA_VISIBLE_DEVICES=0 python3 ./nesd/train.py train \
    --workspace=$WORKSPACE \
    --config_yaml="./kqq_scripts/train/configs/01a.yaml" \
    --gpus=1
   
CUDA_VISIBLE_DEVICES=1 python3 ./nesd/inference.py inference \
    --workspace=$WORKSPACE \
    --config_yaml="./kqq_scripts/train/configs/01a.yaml" \
    --checkpoint_path="/home/tiger/workspaces/nesd2/checkpoints/train/config=01a,gpus=1/step=6000.pth" \
    --gpus=1
   