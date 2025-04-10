#!/bin/bash

scriptDir=src/concept_mapper
fileDir=eraser_movie_dev/layer12/explanation  # Changed to match your explanation directory
classifierDir=eraser_movie/layer12/model/model  # Changed to match where your classifier is saved

# Create output directories
mkdir -p eraser_movie_dev/layer12/explanation/latent_concepts/position_prediction

layer=12
echo "Processing layer ${layer}"
python ${scriptDir}/logistic_regression.py \
    --test_file_path ${fileDir}/cluster_Labels_90%/clusterLabel_layer${layer}.json \
    --layer ${layer} \
    --save_path ./eraser_movie_dev/layer12/explanation/latent_concepts/position_prediction/ \
    --do_predict \
    --classifier_file_path ${classifierDir}/layer_${layer}_classifier.pkl \
    --load_classifier_from_local