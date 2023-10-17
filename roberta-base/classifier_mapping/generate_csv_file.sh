#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --account=def-emilios
#SBATCH --mem=512G 
#SBATCH --mail-type=ALL # 发送哪一种email通知：BEGIN,END,FAIL,ALL
#SBATCH --mail-user=xm863858@dal.ca # 把通知发送到哪一个邮箱

cluster_num=400
clusterDir=../clustering/eraser_movie
data=movie_train.txt

minfreq=5
maxfreq=20
delfreq=1000000

saveDir=clusters_csv_train
mkdir $saveDir

for i in {0..12}
do
echo $i
datasetFile=${clusterDir}/layer$i/$data.tok.sent_len-layer${i}_min_${minfreq}_max_${maxfreq}_del_${delfreq}-dataset.json
python scripts/generate_csv_file.py --dataset_file $datasetFile --cluster_file ${clusterDir}/layer$i/results/clusters-$cluster_num.txt --output_file $saveDir/clusters-map$i.csv
done