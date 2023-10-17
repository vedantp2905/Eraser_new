#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --account=def-hsajjad
#SBATCH --mem=128G
#SBATCH --mail-type=ALL # 发送哪一种email通知：BEGIN,END,FAIL,ALL
#SBATCH --mail-user=xm863858@dal.ca # 把通知发送到哪一个邮箱

Dir=../

for i in {0..12}
do
clusterLabel=${Dir}/cluster_Labels_CLS_dominant_90%/clusterLabel_layer$i.json
echo Layer$i
python scripts/get_stats_of_labeled_clusters.py -c $clusterLabel
done