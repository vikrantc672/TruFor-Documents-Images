#!/usr/bin/env bash
INPUT_DIR=/home/vikrant/Downloads/TruFor-Documents-Images-master/forged_document
OUTPUT_DIR=./output
MASK_RESULT_DIR=./mask
path=$(realpath ${INPUT_DIR})
last_word=$(basename "$path")
mkdir -p ${OUTPUT_DIR}
mkdir -p ${MASK_RESULT_DIR}
# docker run --runtime=nvidia --gpus all -v $(realpath ${INPUT_DIR}):/data -v $(realpath ${OUTPUT_DIR}):/data_out trufor -gpu 0 -in data/ -out data_out/
docker run -v $(realpath ${INPUT_DIR}):/data -v $(realpath ${OUTPUT_DIR}):/data_out trufor -gpu -1 -in data/ -out data_out/

python3 mask.py --image_dir $(realpath ${INPUT_DIR}) --mask_npz_dir $(realpath ${OUTPUT_DIR}) --result_mask_dir $(realpath ${MASK_RESULT_DIR})
python3 onlyscore_csv.py --image_dir $(realpath ${INPUT_DIR}) --output_dir $(realpath ${OUTPUT_DIR}) --score_file ${last_word}
python3 accuracy.py --score_file ${last_word}
