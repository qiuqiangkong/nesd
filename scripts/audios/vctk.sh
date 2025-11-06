#!/bin/bash
python -m audios.vctk \
	--dataset_dir="./datasets/vctk" \
	--segment_duration=2.0 \
	--output_dir="./datasets/vctk_2s_no_silence" \
	--split="train"

python -m audios.vctk \
	--dataset_dir="./datasets/vctk" \
	--segment_duration=2.0 \
	--output_dir="./datasets/vctk_2s_no_silence" \
	--split="test"
