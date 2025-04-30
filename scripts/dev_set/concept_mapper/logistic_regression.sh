#!/bin/bash

layer=6

scriptDir=src/concept_mapper
fileDir=eraser_movie_dev/layer${layer}/representation_info  # Changed to use representation info directory
classifierDir=eraser_movie/layer${layer}/model/model  # Simplified path

# Create output directories
savePath=eraser_movie_dev/layer${layer}/explanation/latent_concepts/position_prediction
mkdir -p ${savePath}

echo "Processing layer ${layer}"

# Debug prints
echo "Input file: ${fileDir}/explanation_words_representation_layer${layer}.csv"
echo "Classifier: ${classifierDir}/layer_${layer}_classifier.pkl"
echo "Save path: ${savePath}"

python ${scriptDir}/logistic_regression.py \
    --test_file_path ${fileDir}/explanation_words_representation_layer${layer}.csv \
    --layer ${layer} \
    --save_path ${savePath}/ \
    --do_predict \
    --classifier_file_path ${classifierDir}/layer_${layer}_classifier.pkl \
    --load_classifier_from_local