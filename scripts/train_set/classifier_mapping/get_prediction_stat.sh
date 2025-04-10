#!/bin/bash

fileDir=eraser_movie/layer12/model/validate_predictions/
scriptDir=src/classifier_mapping

layer=12
python ${scriptDir}/get_prediction_stats.py \
  --layer ${layer} \
  --file_path ${fileDir}

