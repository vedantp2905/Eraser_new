#!/bin/bash

clusterPath=eraser_movie/layer12/results
explanation=eraser_movie/layer12/explanation/explanation_CLS.txt
clusterSize=300
percentage=90

savePath=eraser_movie/layer12/explanation/cluster_Labels_${percentage}%
mkdir -p ${savePath}

layer=12
saveFile=${savePath}/clusterLabel_layer${layer}.json
echo "Layer${layer}"
python src/generate_explanation_files/generate_all_tokens_explanation.py \
    -i ${clusterPath}/clusters-${clusterSize}.txt \
    -e ${explanation} \
    -s ${saveFile}