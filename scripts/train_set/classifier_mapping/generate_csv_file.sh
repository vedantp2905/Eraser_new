#!/bin/bash

cluster_num=300
clusterDir=eraser_movie
data=movie_train.txt
scriptDir=src/classifier_mapping

minfreq=0
maxfreq=15000
delfreq=15000

saveDir=eraser_movie/layer12/clusters_csv_train
mkdir $saveDir

i=12
datasetFile=${clusterDir}/layer$i/$data.tok.sent_len-layer${i}_min_${minfreq}_max_${maxfreq}_del_${delfreq}-dataset.json
python ${scriptDir}/generate_csv_file.py --dataset_file $datasetFile --cluster_file ${clusterDir}/layer$i/results/clusters-$cluster_num.txt --output_file $saveDir/clusters-map$i.csv
