#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --account=def-hsajjad
#SBATCH --mem=128G 
#SBATCH --mail-type=ALL # 发送哪一种email通知：BEGIN,END,FAIL,ALL
#SBATCH --mail-user=xm863858@dal.ca # 把通知发送到哪一个邮箱
clusterPath=../../clustering/eraser_movie
explanation=explanation_words.txt
clusterSize=400
percentage=90

savePath=../cluster_Labels_$percentage%
mkdir $savePath

for i in {0..12}
do
saveFile=${savePath}/clusterLabel_layer$i.json
echo Layer$i
python scripts/generate_cluster_label_all_tokens.py -c $clusterPath/layer$i/results/clusters-$clusterSize.txt -e $explanation -p $percentage -s $saveFile
done