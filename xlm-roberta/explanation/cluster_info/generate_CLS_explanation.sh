#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --account=def-emilios
#SBATCH --mem=512G 
#SBATCH --mail-type=ALL # 发送哪一种email通知：BEGIN,END,FAIL,ALL
#SBATCH --mail-user=xm863858@dal.ca # 把通知发送到哪一个邮箱

scriptDir=scripts
model=../../glue-eraser-movie-xlm-roberta-base
inputPath=../../../data # path to a sentence file

saveDir=.

layer=12
python ${scriptDir}/generate_CLS_explanation.py --dataset-name-or-path ${inputPath}/movie_train.json --task-name sst2 --model-name ${model} --tokenizer-name ${model} --save-dir ${saveDir}

