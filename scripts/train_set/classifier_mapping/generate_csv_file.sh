#!/bin/bash

cluster_num=400
clusterDir=../clustering/eraser_movie
data=movie_train.txt
scriptDir=../../../src/classifier_mapping

minfreq=5
maxfreq=20
delfreq=1000000

saveDir=clusters_csv_train
mkdir $saveDir

i=12
datasetFile=${clusterDir}/layer$i/$data.tok.sent_len-layer${i}_min_${minfreq}_max_${maxfreq}_del_${delfreq}-dataset.json
python ${scriptDir}/generate_csv_file.py --dataset_file $datasetFile --cluster_file ${clusterDir}/layer$i/results/clusters-$cluster_num.txt --output_file $saveDir/clusters-map$i.csv
