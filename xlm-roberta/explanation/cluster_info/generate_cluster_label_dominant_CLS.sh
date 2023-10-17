#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --account=def-emilios
#SBATCH --mem=128G
#SBATCH --mail-type=ALL # 发送哪一种email通知：BEGIN,END,FAIL,ALL
#SBATCH --mail-user=xm863858@dal.ca # 把通知发送到哪一个邮箱

scriptDir=scripts
clusterPath=../../clustering/eraser_movie
clusterSize=400
percentage=90
explanation=explanation_CLS.txt

savePath=../cluster_Labels_CLS_dominant_$percentage%
mkdir $savePath

for layer in {0..12}
do
echo BERT-base-layer${layer}
saveFile=${savePath}/clusterLabel_layer$layer.json
python ${scriptDir}/generate_cluster_label_dominant_CLS.py -c $clusterPath/layer$layer/results/clusters-$clusterSize.txt -e $explanation -p $percentage -s $saveFile 
done
