#!/bin/bash

# Accept layer argument
layer=${1}

scriptDir=src/concept_mapper
input=eraser_movie_dev/layer${layer}/movie_dev_subset.txt
working_file=${input}.tok.sent_len

dataPath=eraser_movie_dev
minfreq=0
maxfreq=10000000
delfreq=10000000

savePath=eraser_movie_dev/layer${layer}/representation_info
mkdir -p ${savePath}

saveFile=${savePath}/explanation_words_representation_layer${layer}.csv
explanation=eraser_movie_dev/IG_explanation_files_mass_50/explanation_layer_6.txt

python ${scriptDir}/match_representation.py \
    --datasetFile ${working_file}-layer${layer}_min_${minfreq}_max_${maxfreq}_del_${delfreq}-dataset.json \
    --explanationFile ${explanation} \
    --outputFile ${saveFile}
