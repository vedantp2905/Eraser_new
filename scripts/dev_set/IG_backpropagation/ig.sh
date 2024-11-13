#!/bin/bash

scriptDir=../../../src/IG_backpropagation
inputFile=../clustering/movie_dev_subset.txt.tok.sent_len
model=xxx # add path to the model to "xxx"

outDir=../IG_attributions
mkdir ${outDir}

layer=12
saveFile=${outDir}/IG_explanation_layer_${layer}.csv
python ${scriptDir}/ig_for_sequence_classification.py ${inputFile} ${model} $layer ${saveFile}