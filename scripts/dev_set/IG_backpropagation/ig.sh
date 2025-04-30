#!/bin/bash

scriptDir=src/IG_backpropagation
inputFile=eraser_movie_dev/movie_dev_subset.txt.tok.sent_len
model="/lustre/hdd/LAS/jannesar-lab/vedant29/finetuned_models/codebert-pos-lang-classification/best_model"

outDir=eraser_movie_dev/IG_attributions
mkdir -p ${outDir}

# Loop over layers 0 to 12
for layer in {6..6}; do
    saveFile=${outDir}/IG_explanation_layer_${layer}.csv

    echo "Running for layer ${layer}..."
    python ${scriptDir}/ig_for_sequence_classification.py \
        --input_file ${inputFile} \
        --model ${model} \
        --layer ${layer} \
        --save_file ${saveFile} \
        --model_type roberta 
done