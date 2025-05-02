#!/bin/bash

# Accept layer argument
layer=${1}

scriptDir=src/generate_explanation_files
model=${2}
inputFile=eraser_movie_dev/movie_dev_subset.txt.tok

saveDir=eraser_movie_dev/layer${layer}/explanation
mkdir -p ${saveDir}    # Fixed: Changed {$saveDir} to ${saveDir}

python ${scriptDir}/generate_CLS_explanation.py \
    --dataset-name-or-path ${inputFile} \
    --model-name ${model} \
    --tokenizer-name ${model} \
    --save-dir ${saveDir} \
    --batch-size 16 \
    --cpu-only  # Add this if GPU memory is still an issue