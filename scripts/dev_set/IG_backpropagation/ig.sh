#!/bin/bash

scriptDir=src/IG_backpropagation
inputFile=eraser_movie_dev/movie_dev_subset.txt.tok.sent_len
model="google-bert/bert-base-cased"

outDir=eraser_movie_dev/IG_attributions
mkdir -p ${outDir}

layer=12
saveFile=${outDir}/IG_explanation_layer_${layer}.csv

# Change to use RobertaForSequenceClassification instead of BertForSequenceClassification
python ${scriptDir}/ig_for_sequence_classification.py \
    --input_file ${inputFile} \
    --model ${model} \
    --layer ${layer} \
    --save_file ${saveFile} \
    --model_type bert