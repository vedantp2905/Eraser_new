#!/bin/bash

scriptDir=src/generate_explanation_files
model="best_codebert_model"
inputFile=eraser_movie/movie_train.txt.tok

saveDir=eraser_movie/layer12/explanation
mkdir -p ${saveDir}    # Fixed: Changed {$saveDir} to ${saveDir}

layer=12
python ${scriptDir}/generate_CLS_explanation.py \
    --dataset-name-or-path ${inputFile} \
    --model-name ${model} \
    --tokenizer-name ${model} \
    --save-dir ${saveDir}