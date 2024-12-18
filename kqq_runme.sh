
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

python3 nesd/evaluate_dcase2021_task3.py plot_predict --task_type="dcase2019"

python3 nesd/evaluate_dcase2021_task3.py process_mat_write_csv \
    --task_type="dcase2019" \
    --csv="./submissions/01_test/fold6_room1_mix001.csv"

python3 nesd/evaluate_yin.py

#
python3 nesd/d19t3_oracle_pred.py

python3 nesd/test6.py

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
    
######### DCASE2019 Task3 SED
CUDA_VISIBLE_DEVICES=2 python3 ./nesd/train_sed.py train \
    --workspace=$WORKSPACE \
    --config_yaml="./kqq_scripts/train/configs/sed_dcase2019_task3_01a.yaml" \
    --gpus=1 

CUDA_VISIBLE_DEVICES=2 python3 ./nesd/inference_sed.py inference_dcase2021_single_map \
    --workspace=$WORKSPACE \
    --config_yaml="./kqq_scripts/train/configs/sed_dcase2019_task3_01a.yaml" \
    --checkpoint_path="/home/tiger/workspaces/nesd2/checkpoints/train_sed/config=sed_dcase2019_task3_01a,gpus=1/step=10000.pth" \
    --gpus=1

######### DCASE2019 Task3 SED on freezed Loc model
CUDA_VISIBLE_DEVICES=2 python3 ./nesd/train_sed_fz_loc.py train \
    --workspace=$WORKSPACE \
    --config_yaml="./kqq_scripts/train/configs/sed_fz_loc_dcase2019_task3_01a.yaml" \
    --gpus=1

CUDA_VISIBLE_DEVICES=2 python3 ./nesd/inference_sed_fz_loc.py inference_dcase2021_single_map \
    --workspace=$WORKSPACE \
    --config_yaml="./kqq_scripts/train/configs/sed_fz_loc_dcase2019_task3_01a.yaml" \
    --checkpoint_path="/home/tiger/workspaces/nesd2/checkpoints/train_sed_fz_loc/config=sed_fz_loc_dcase2019_task3_01a,gpus=1/step=10000.pth" \
    --gpus=1


###########################
python nesd/test_room.py

############# 2023.10 New ############

WORKSPACE="/home/qiuqiangkong/workspaces/nesd2"

python nesd/test5.py    # latest simulator
python nesd/test_dataloader.py  # render iss data and plot simulator
python nesd/test_plot.py    # Plot data simulator

# Prepare data
python nesd/process_vctk_dataset.py

python ./nesd/dataset_creation/pack_audios_to_hdf5s/dcase2016_task2.py

python ./nesd/dataset_creation/pack_audios_to_hdf5s/dcase2018_task2.py

python ./nesd/dataset_creation/pack_audios_to_hdf5s/fsd50k.py

python ./nesd/dataset_creation/pack_audios_to_hdf5s/musdb18hq.py \
    --dataset_dir="/home/qiuqiangkong/datasets/musdb18hq" \
    --split="${SPLIT}" \
    --hdf5s_dir="${WORKSPACE}/hdf5s/musdb18hq/${SPLIT}" \
    --sample_rate=24000 \
    --segment_seconds=2.0

###
python nesd/prepare_paths_dict_d22.py

### --- pyroomacoustics ---
# train
CUDA_VISIBLE_DEVICES=0 python nesd/train2.py train \
    --workspace=$WORKSPACE \
    --config_yaml="./kqq_scripts/train/configs2/01a.yaml"

# Inference
CUDA_VISIBLE_DEVICES=0 python nesd/inference2.py inference \
    --workspace="" \
    --config_yaml="./kqq_scripts/train/configs2/01a.yaml" \
    --checkpoint_path=""

python nesd/inference2.py add

# Inference DCASE2019 Task3
CUDA_VISIBLE_DEVICES=0 python nesd/inference2_dcase2019_task3.py inference \
    --workspace="" \
    --config_yaml="./kqq_scripts/train/configs2/02a.yaml" \
    --checkpoint_path=""

CUDA_VISIBLE_DEVICES=0 python nesd/inference2_dcase2019_task3.py plot

### --- Freefield ---
WORKSPACE="/home/qiuqiangkong/workspaces/nesd2"
CUDA_VISIBLE_DEVICES=0 python nesd/train2_freefield.py train \
    --workspace=$WORKSPACE \
    --config_yaml="./kqq_scripts/train/configs2/freefield_01a.yaml"


### --- dcase 2019 task 3 ---
# Train
SPLIT="train"
python nesd/dataset_creation/pack_audios_to_hdf5s/dcase2019_task3.py \
    --dataset_dir="/home/qiuqiangkong/datasets/dcase2019/task3/downloaded_package" \
    --split=$SPLIT \
    --hdf5s_dir="${WORKSPACE}/hdf5s/dcase2019_task3/${SPLIT}" \
    --sample_rate=24000

WORKSPACE="/home/qiuqiangkong/workspaces/nesd2"
CUDA_VISIBLE_DEVICES=0 python nesd/train2_dcase2019_task3.py train \
    --workspace=$WORKSPACE \
    --config_yaml="./kqq_scripts/train/configs2/dcase2019_task3_01a.yaml"

### --- dcase 2020 task 3 ---
SPLIT="train"
python nesd/dataset_creation/pack_audios_to_hdf5s/dcase2020_task3.py \
    --dataset_dir="/home/qiuqiangkong/datasets/dcase2020/task3" \
    --split=$SPLIT \
    --hdf5s_dir="${WORKSPACE}/hdf5s/dcase2020_task3/${SPLIT}" \
    --sample_rate=24000

### --- dcase 2021 task 3 ---
SPLIT="train"
python nesd/dataset_creation/pack_audios_to_hdf5s/dcase2021_task3.py \
    --dataset_dir="/home/qiuqiangkong/datasets/dcase2021/task3/downloaded_package" \
    --split=$SPLIT \
    --hdf5s_dir="${WORKSPACE}/hdf5s/dcase2021_task3/${SPLIT}" \
    --sample_rate=24000


### --- dcase 2022 task 3 ---
SPLIT="train"
python nesd/dataset_creation/pack_audios_to_hdf5s/dcase2022_task3.py \
    --dataset_dir="/home/qiuqiangkong/datasets/dcase2022/task3" \
    --split=$SPLIT \
    --hdf5s_dir="${WORKSPACE}/hdf5s/dcase2022_task3/${SPLIT}" \
    --sample_rate=24000

### --- dcase 2023 task 3 ---
SPLIT="train"
python nesd/dataset_creation/pack_audios_to_hdf5s/dcase2023_task3.py \
    --dataset_dir="/home/qiuqiangkong/datasets/dcase2023/task3" \
    --split=$SPLIT \
    --hdf5s_dir="${WORKSPACE}/hdf5s/dcase2023_task3/${SPLIT}" \
    --sample_rate=24000

### --- concatenate datasets ---
WORKSPACE="/home/qiuqiangkong/workspaces/nesd2"
CUDA_VISIBLE_DEVICES=1 python nesd/train2_concat.py train \
    --workspace=$WORKSPACE \
    --config_yaml="./kqq_scripts/train/configs2/07a.yaml"


#########################
# Inference dcase 2019 task 3
CUDA_VISIBLE_DEVICES=2 python nesd/inference2_dcase2019_task3.py inference \
    --workspace="" \
    --config_yaml="./kqq_scripts/train/configs2/dcase2019_task3_01a.yaml" \
    --checkpoint_path="/home/qiuqiangkong/workspaces/nesd2/checkpoints/train2_dcase2019_task3/config=dcase2019_task3_01a/epoch=45-step=46000-test_loss=0.045.ckpt"

python nesd/inference2_dcase2019_task3.py plot

ffmpeg -framerate 10 -i '_tmp/_zz_%04d.png' -r 30 -pix_fmt yuv420p 123.mp4

python nesd/inference2_dcase2019_task3.py center_to_csv

# Inference dcase 2020 task 3

CUDA_VISIBLE_DEVICES=2 python nesd/inference2_dcase2020_task3.py inference \
    --workspace="" \
    --config_yaml="./kqq_scripts/train/configs2/03a.yaml" \
    --checkpoint_path="/home/qiuqiangkong/workspaces/nesd2/checkpoints/train2/config=03a/epoch=23-step=24000-test_loss=0.031.ckpt"
python nesd/inference2_dcase2020_task3.py plot

CUDA_VISIBLE_DEVICES=2 python nesd/inference2_dcase2020_task3.py inference_depth     --workspace=""     --config_yaml="./kqq_scripts/train/configs2/05a.yaml"     --checkpoint_path="/home/qiuqiangkong/workspaces/nesd2/checkpoints/train2/config=05a/epoch=34-step=35000-test_loss=0.025.ckpt"
python nesd/inference2_dcase2020_task3.py plot_depth

CUDA_VISIBLE_DEVICES=2 python nesd/inference2_dcase2020_task3.py inference_sep     --workspace=""     --config_yaml="./kqq_scripts/train/configs2/06a.yaml"     --checkpoint_path="/home/qiuqiangkong/workspaces/nesd2/checkpoints/train2/config=06a/epoch=49-step=50000-test_loss=0.129.ckpt"

# Inference dcase 2022 task 3
CUDA_VISIBLE_DEVICES=2 python nesd/inference2_dcase2022_task3.py inference \
    --workspace="" \
    --config_yaml="./kqq_scripts/train/configs2/03a.yaml" \
    --checkpoint_path="/home/qiuqiangkong/workspaces/nesd2/checkpoints/train2/config=03a/epoch=23-step=24000-test_loss=0.031.ckpt"
python nesd/inference2_dcase2022_task3.py plot

CUDA_VISIBLE_DEVICES=2 python nesd/inference2_dcase2022_task3.py inference_cla \
    --workspace="" \
    --config_yaml="./kqq_scripts/train/configs2/15a.yaml" \
    --checkpoint_path="/home/qiuqiangkong/workspaces/nesd2/checkpoints/train2/config=03a/epoch=23-step=24000-test_loss=0.031.ckpt"
python nesd/inference2_dcase2022_task3.py plot_cla

######## Create RIR
python nesd/test_rir.py

python nesd/dataset_creation/pack_audios_to_hdf5s/tau_noise.py \
    --dataset_dir="/home/qiuqiangkong/datasets/dcase2022/tau-srir/TAU-SNoise_DB" \
    --split="train" \
    --hdf5s_dir="/home/qiuqiangkong/workspaces/nesd2/hdf5s/tau-noise" \
    --sample_rate=24000

####
# write rigid IR to h5
python nesd2/rigid.py