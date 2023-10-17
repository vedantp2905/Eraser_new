#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --account=def-emilios
#SBATCH --mem=128G
#SBATCH --mail-type=ALL # 发送哪一种email通知：BEGIN,END,FAIL,ALL
#SBATCH --mail-user=xm863858@dal.ca # 把通知发送到哪一个邮箱

filePath='clusters_csv_train'
scriptsDir=scripts

for layer in {0..12}
do
echo BERT-base-layer${layer}
python ${scriptsDir}/check_cluster_idx_CLS.py --filename ${filePath}/clusters-map${layer}.csv
done