#!/bin/bash

scriptDir=../../../src/generate_explanation_files
inputDir=../IG_attributions
outDir=../IG_explanation_files_mass_50

mkdir ${outDir}

layer=12
echo ${inputDir}/IG_explanation_layer_${layer}.csv
saveFile=${outDir}/explanation_layer_${layer}.txt
python ${scriptDir}/generate_IG_explanation_salient_words.py ${inputDir}/IG_explanation_layer_${layer}.csv ${saveFile} top-k --attribution_mass 0.5