#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --account=def-hsajjad
#SBATCH --mem=128G 
#SBATCH --mail-type=ALL # 发送哪一种email通知：BEGIN,END,FAIL,ALL
#SBATCH --mail-user=xm863858@dal.ca # 把通知发送到哪一个邮箱

scriptDir=scripts
clusterPath=../../clustering/eraser_movie
clusterSize=400

savePath=.

explanation=explanation_CLS.txt
saveFile=$savePath/explanation_words.txt

layer=12
python ${scriptDir}/generate_all_tokens_explanation.py -c $clusterPath/layer$layer/results/clusters-$clusterSize.txt -e $explanation -s $saveFile

