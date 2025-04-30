#!/bin/bash

layer=6
fileDir=eraser_movie/layer${layer}/model/validate_predictions/
scriptDir=src/classifier_mapping

layer=6
python ${scriptDir}/get_prediction_stats.py \
  --layer ${layer} \
  --file_path ${fileDir}

