#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --account=def-hsajjad
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G 
#SBATCH --mail-type=ALL # 发送哪一种email通知：BEGIN,END,FAIL,ALL
#SBATCH --mail-user=xm863858@dal.ca # 把通知发送到哪一个邮箱

inputFile=../clustering/movie_train.txt.tok.sent_len
model=../glue-eraser-movie-roberta-base-cased

outDir=../IG_attributions
mkdir ${outDir}

layer=0
saveFile=${outDir}/IG_explanation_layer_${layer}.csv
python ig_for_sequence_classification.py ${inputFile} ${model} $layer ${saveFile}