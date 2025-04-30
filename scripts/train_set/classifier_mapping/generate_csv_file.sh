#!/bin/bash

layer=6
cluster_num=30
clusterDir=eraser_movie
data=movie_train.txt
scriptDir=src/classifier_mapping

minfreq=0
maxfreq=10000000
delfreq=10000000

saveDir=eraser_movie/layer${layer}/clusters_csv_train
mkdir $saveDir

i=6
datasetFile=${clusterDir}/layer$i/$data.tok.sent_len-layer${i}_min_${minfreq}_max_${maxfreq}_del_${delfreq}-dataset.json
python ${scriptDir}/generate_csv_file.py --dataset_file $datasetFile --cluster_file ${clusterDir}/layer$i/results/clusters-$cluster_num.txt --output_file $saveDir/clusters-map$i.csv
