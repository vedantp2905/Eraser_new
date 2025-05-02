#!/bin/bash

scriptDir=src/generate_explanation_files
inputDir=eraser_movie/IG_attributions
outDir=eraser_movie/IG_explanation_files_mass_50

mkdir ${outDir}

# Accept layer argument
layer=${1}

echo ${inputDir}/IG_explanation_layer_${layer}.csv
saveFile=${outDir}/explanation_layer_${layer}.txt
python ${scriptDir}/generate_IG_explanation_salient_words.py ${inputDir}/IG_explanation_layer_${layer}.csv ${saveFile} top-k --attribution_mass 0.5