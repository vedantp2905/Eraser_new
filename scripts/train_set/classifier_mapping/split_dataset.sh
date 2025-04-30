#!/bin/bash

scriptDir=src/classifier_mapping
dirName=eraser_movie
layer=6
baseDir=${dirName}/layer${layer}

saveDir=${baseDir}/split_dataset
mkdir -p ${saveDir}

filePath=${baseDir}/clusters_csv_train/

python ${scriptDir}/split_dataset.py \
  --file_path ${filePath} \
  --layer ${layer} \
  --validation_size 0.1 \
  --train_dataset_save_path ${saveDir}/train/ \
  --validation_dataset_save_path ${saveDir}/validation/ \
  --id_save_filename ${saveDir}/id.txt \
  --is_first_file

