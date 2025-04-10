#!/bin/bash

scriptDir=src/classifier_mapping
baseDir=eraser_movie/layer12
fileDir=${baseDir}/split_dataset
savePath=${baseDir}/model


layer=12
python ${scriptDir}/logistic_regression.py \
    --train_file_path ${fileDir}/train/train_df_${layer}.csv \
    --validate_file_path ${fileDir}/validation/validation_df_${layer}.csv \
    --layer ${layer} \
    --save_path ${savePath} \
    --do_train \
    --do_validate
