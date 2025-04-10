#!/bin/bash

scriptDir=src/concept_mapper
input=eraser_movie_dev/layer12/movie_dev_subset.txt
working_file=${input}.tok.sent_len

dataPath=eraser_movie_dev
minfreq=0
maxfreq=15000
delfreq=15000

savePath=eraser_movie_dev/layer12/representation_info
mkdir -p ${savePath}

layer=12
saveFile=${savePath}/explanation_words_representation_layer${layer}.csv
explanation=eraser_movie_dev/layer12/explanation/explanation_CLS.txt

python ${scriptDir}/match_representation.py \
    --datasetFile ${working_file}-layer${layer}_min_${minfreq}_max_${maxfreq}_del_${delfreq}-dataset.json \
    --explanationFile ${explanation} \
    --outputFile ${saveFile}
