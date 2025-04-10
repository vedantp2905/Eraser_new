#!/bin/bash


scriptDir=src/concept_mapper
fileDir=position_representation_info  #saliency_representation_info #
classifierDir=../../train_set/classifier_mapping/result/model

mkdir "latent_concepts"
mkdir "latent_concepts/position_prediction"


layer=12
echo ${layer}
python ${scriptDir}/logistic_regression.py \
  --test_file_path ${fileDir}/explanation_words_representation_layer${layer}.csv \
  --layer ${layer} \
  --save_path ./latent_concepts/position_prediction/ \
  --do_predict \
  --classifier_file_path $classifierDir/layer_${layer}_classifier.pkl \
  --load_classifier_from_local