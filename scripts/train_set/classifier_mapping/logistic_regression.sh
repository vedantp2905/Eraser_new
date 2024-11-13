#!/bin/bash

scriptDir=../../../src/classifier_mapping
fileDir=split_dataset
savePath=result


layer=12
python ${scriptDir}/logistic_regression.py --train_file_path ${fileDir}/train/train_df_${layer}.csv --validate_file_path ${fileDir}/validation/validation_df_${layer}.csv --layer ${layer} --save_path ${savePath} --do_train --do_validate
