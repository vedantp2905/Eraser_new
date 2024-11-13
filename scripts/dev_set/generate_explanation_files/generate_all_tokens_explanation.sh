#!/bin/bash

clusterPath=../clustering/eraser_movie_dev
explanation=explanation_words.txt
clusterSize=400
percentage=90

savePath=../cluster_Labels_$percentage%
mkdir $savePath

layer=12
saveFile=${savePath}/clusterLabel_layer$layer.json
echo Layer$layer
python scripts/generate_cluster_label_all_tokens.py -c $clusterPath/layer$layer/results/clusters-$clusterSize.txt -e $explanation -p $percentage -s $saveFile
