#!/bin/bash

# Accept layer argument
layer=${1}
model=${2}

scriptDir=src/IG_backpropagation
inputFile=eraser_movie_dev/movie_dev_subset.txt.tok.sent_len

outDir=eraser_movie_dev/IG_attributions
mkdir -p ${outDir}

saveFile=${outDir}/IG_explanation_layer_${layer}.csv

echo "Running for layer ${layer}..."
python ${scriptDir}/ig_for_sequence_classification.py \
    --input_file ${inputFile} \
    --model ${model} \
    --layer ${layer} \
    --save_file ${saveFile} \
    --model_type roberta 