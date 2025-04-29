#!/bin/bash

scriptDir=src/generate_explanation_files
model="/lustre/hdd/LAS/jannesar-lab/vedant29/Eraser_new/bert-cased-finetuned"
inputFile=eraser_movie_dev/movie_dev_subset.txt.tok

saveDir=eraser_movie_dev/layer12/explanation
mkdir -p ${saveDir}    # Fixed: Changed {$saveDir} to ${saveDir}

layer=12
python ${scriptDir}/generate_CLS_explanation.py \
    --dataset-name-or-path ${inputFile} \
    --model-name ${model} \
    --tokenizer-name ${model} \
    --save-dir ${saveDir} \
    --batch-size 16 \
    --cpu-only  # Add this if GPU memory is still an issue