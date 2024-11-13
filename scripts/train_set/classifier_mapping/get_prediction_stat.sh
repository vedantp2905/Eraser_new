#!/bin/bash

fileDir=result/validate_predictions/
scriptDir=../../../src/classifier_mapping

layer=12
python get_prediction_stats.py \
  --layer ${layer} \
  --file_path ${fileDir}

