#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --account=def-hsajjad
#SBATCH --mem=128G 
#SBATCH --mail-type=ALL # 发送哪一种email通知：BEGIN,END,FAIL,ALL
#SBATCH --mail-user=xm863858@dal.ca # 把通知发送到哪一个邮箱

clusterPath=../../clustering/eraser_movie
clusterSize=400
percentage=90

for i in {0..12}
do
explanation=../cluster_info/explanation_CLS.txt
clusterLabel=../cluster_Labels_$percentage%/clusterLabel_layer$i.json
echo Layer$i
python Method1_eval.py -c $clusterPath/layer$i/results/clusters-$clusterSize.txt -e $explanation -l $clusterLabel -p $percentage
done
