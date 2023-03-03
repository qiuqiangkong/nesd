
WORKSPACE="/home/tiger/workspaces/nesd2"

SAMPLE_RATE=24000

######## VCTK free field
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


######## DCASE 2019
SPLIT="test"
python3 ./nesd/dataset_creation/pack_audios_to_hdf5s/dcase2019_task3.py \
    --dataset_dir="/home/tiger/datasets/dcase2019/task3" \
    --split=$SPLIT \
    --hdf5s_dir="${WORKSPACE}/hdf5s/dcase2019_task3/sr=${SAMPLE_RATE}/${SPLIT}" \
    --sample_rate=$SAMPLE_RATE

CUDA_VISIBLE_DEVICES=3 python3 ./nesd/train.py train \
    --workspace=$WORKSPACE \
    --config_yaml="./kqq_scripts/train/configs/dcase2019_task3_01a.yaml" \
    --gpus=1

CUDA_VISIBLE_DEVICES=0 python3 ./nesd/inference.py inference_dcase2021_single_map \
    --workspace=$WORKSPACE \
    --config_yaml="./kqq_scripts/train/configs/dcase2019_task3_01a.yaml" \
    --checkpoint_path="/home/tiger/workspaces/nesd2/checkpoints/train/config=dcase2019_task3_01a,gpus=1/step=10000.pth" \
    --gpus=1

   
######## DCASE 2021
for SPLIT in "train" "test"
do  
    python3 ./nesd/dataset_creation/pack_audios_to_hdf5s/dcase2021_task3.py \
        --dataset_dir="/home/tiger/datasets/dcase2021/task3" \
        --split=$SPLIT \
        --hdf5s_dir="${WORKSPACE}/hdf5s/dcase2021_task3/sr=${SAMPLE_RATE}/${SPLIT}" \
        --sample_rate=$SAMPLE_RATE
done

CUDA_VISIBLE_DEVICES=6 python3 ./nesd/train.py train \
    --workspace=$WORKSPACE \
    --config_yaml="./kqq_scripts/train/configs/dcase2021_task3_01a.yaml" \
    --gpus=1

CUDA_VISIBLE_DEVICES=0 python3 ./nesd/inference.py inference_dcase2021_single_map \
    --workspace=$WORKSPACE \
    --config_yaml="./kqq_scripts/train/configs/dcase2021_task3_03c.yaml" \
    --checkpoint_path="/home/tiger/workspaces/nesd2/checkpoints/train/config=dcase2021_task3_03c,gpus=1/step=10000.pth" \
    --gpus=1

ffmpeg -framerate 10 -i '_tmp/_zz_%03d.jpg' -r 30 -pix_fmt yuv420p 123.mp4

python3 nesd/evaluate_dcase2021_task3.py process_mat_write_csv --csv="./submissions/01_test/fold6_room1_mix001.csv"

python3 nesd/evaluate_yin.py

######## DCASE 2022
SPLIT="train"
python3 ./nesd/dataset_creation/pack_audios_to_hdf5s/dcase2022_task3.py \
    --dataset_dir="/home/tiger/datasets/dcase2022/task3" \
    --split=$SPLIT \
    --hdf5s_dir="${WORKSPACE}/hdf5s/dcase2022_task3/sr=${SAMPLE_RATE}/${SPLIT}" \
    --sample_rate=$SAMPLE_RATE

CUDA_VISIBLE_DEVICES=7 python3 ./nesd/train.py train \
    --workspace=$WORKSPACE \
    --config_yaml="./kqq_scripts/train/configs/dcase2022_task3_01a.yaml" \
    --gpus=1

CUDA_VISIBLE_DEVICES=3 python3 ./nesd/inference.py inference_dcase2021_single_map \
    --workspace=$WORKSPACE \
    --config_yaml="./kqq_scripts/train/configs/dcase2022_task3_01a.yaml" \
    --checkpoint_path="/home/tiger/workspaces/nesd2/checkpoints/train/config=dcase2022_task3_01a,gpus=1/step=10000.pth" \
    --gpus=1

#### pack dcase2018task2, music data
python3 ./nesd/dataset_creation/pack_audios_to_hdf5s/dcase2018_task2.py \
    --dataset_dir="/home/tiger/datasets/dcase2018/task2/dataset_root/" \
    --split="${SPLIT}" \
    --hdf5s_dir="${WORKSPACE}/hdf5s/dcase2018_task2/sr=${SAMPLE_RATE}/${SPLIT}" \
    --sample_rate=$SAMPLE_RATE \
    --segment_seconds=3.0

python3 ./nesd/dataset_creation/pack_audios_to_hdf5s/musdb18hq.py \
    --dataset_dir="/home/tiger/datasets/musdb18hq" \
    --split="${SPLIT}" \
    --hdf5s_dir="${WORKSPACE}/hdf5s/musdb18hq/sr=${SAMPLE_RATE}/${SPLIT}" \
    --sample_rate=$SAMPLE_RATE \
    --segment_seconds=3.0
    