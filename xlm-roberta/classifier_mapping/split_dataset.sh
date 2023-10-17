#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --account=def-hsajjad
#SBATCH --mem=512G 
#SBATCH --mail-type=ALL # 发送哪一种email通知：BEGIN,END,FAIL,ALL
#SBATCH --mail-user=xm863858@dal.ca # 把通知发送到哪一个邮箱

scriptsDir=scripts

saveDir='split_dataset' #'split_dataset_CLS'
mkdir ${saveDir}

filePath='clusters_csv_train/'

python ${scriptsDir}/split_dataset.py \
  --file_path ${filePath} \
  --layer 0 \
  --validation_size 0.1 \
  --train_dataset_save_path ${saveDir}/train/ \
  --validation_dataset_save_path ${saveDir}/validation/ \
  --id_save_filename ${saveDir}/id.txt \
  --is_first_file \
#  --only_CLS

python ${scriptsDir}/split_dataset.py \
  --file_path ${filePath} \
  --layer 1 \
  --validation_size 0.1 \
  --train_dataset_save_path ${saveDir}/train/ \
  --validation_dataset_save_path ${saveDir}/validation/ \
  --id_save_filename ${saveDir}/id.txt \
#  --only_CLS

python ${scriptsDir}/split_dataset.py \
  --file_path ${filePath} \
  --layer 2 \
  --validation_size 0.1 \
  --train_dataset_save_path ${saveDir}/train/ \
  --validation_dataset_save_path ${saveDir}/validation/ \
  --id_save_filename ${saveDir}/id.txt \
#  --only_CLS

python ${scriptsDir}/split_dataset.py \
  --file_path ${filePath} \
  --layer 3 \
  --validation_size 0.1 \
  --train_dataset_save_path ${saveDir}/train/ \
  --validation_dataset_save_path ${saveDir}/validation/ \
  --id_save_filename ${saveDir}/id.txt \
#  --only_CLS

python ${scriptsDir}/split_dataset.py \
  --file_path ${filePath} \
  --layer 4 \
  --validation_size 0.1 \
  --train_dataset_save_path ${saveDir}/train/ \
  --validation_dataset_save_path ${saveDir}/validation/ \
  --id_save_filename ${saveDir}/id.txt \
#  --only_CLS

python ${scriptsDir}/split_dataset.py \
  --file_path ${filePath} \
  --layer 5 \
  --validation_size 0.1 \
  --train_dataset_save_path ${saveDir}/train/ \
  --validation_dataset_save_path ${saveDir}/validation/ \
  --id_save_filename ${saveDir}/id.txt \
#  --only_CLS

python ${scriptsDir}/split_dataset.py \
  --file_path ${filePath} \
  --layer 6 \
  --validation_size 0.1 \
  --train_dataset_save_path ${saveDir}/train/ \
  --validation_dataset_save_path ${saveDir}/validation/ \
  --id_save_filename ${saveDir}/id.txt \
#  --only_CLS

python ${scriptsDir}/split_dataset.py \
  --file_path ${filePath} \
  --layer 7 \
  --validation_size 0.1 \
  --train_dataset_save_path ${saveDir}/train/ \
  --validation_dataset_save_path ${saveDir}/validation/ \
  --id_save_filename ${saveDir}/id.txt \
#  --only_CLS

python ${scriptsDir}/split_dataset.py \
  --file_path ${filePath} \
  --layer 8 \
  --validation_size 0.1 \
  --train_dataset_save_path ${saveDir}/train/ \
  --validation_dataset_save_path ${saveDir}/validation/ \
  --id_save_filename ${saveDir}/id.txt \
#  --only_CLS

python ${scriptsDir}/split_dataset.py \
  --file_path ${filePath} \
  --layer 9 \
  --validation_size 0.1 \
  --train_dataset_save_path ${saveDir}/train/ \
  --validation_dataset_save_path ${saveDir}/validation/ \
  --id_save_filename ${saveDir}/id.txt \
#  --only_CLS

python ${scriptsDir}/split_dataset.py \
  --file_path ${filePath} \
  --layer 10 \
  --validation_size 0.1 \
  --train_dataset_save_path ${saveDir}/train/ \
  --validation_dataset_save_path ${saveDir}/validation/ \
  --id_save_filename ${saveDir}/id.txt \
#  --only_CLS

python ${scriptsDir}/split_dataset.py \
  --file_path ${filePath} \
  --layer 11 \
  --validation_size 0.1 \
  --train_dataset_save_path ${saveDir}/train/ \
  --validation_dataset_save_path ${saveDir}/validation/ \
  --id_save_filename ${saveDir}/id.txt \
#  --only_CLS

python ${scriptsDir}/split_dataset.py \
  --file_path ${filePath} \
  --layer 12 \
  --validation_size 0.1 \
  --train_dataset_save_path ${saveDir}/train/ \
  --validation_dataset_save_path ${saveDir}/validation/ \
  --id_save_filename ${saveDir}/id.txt \
#  --only_CLS