#!/bin/bash

# Accept layer argument
layer=${1}
fileDir=eraser_movie/layer${layer}/model/validate_predictions/
scriptDir=src/classifier_mapping

python ${scriptDir}/get_prediction_stats.py \
  --layer ${layer} \
  --file_path ${fileDir}
