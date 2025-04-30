#!/bin/bash

layer=6
clusterPath=eraser_movie/layer${layer}/results
explanation=eraser_movie/layer${layer}/explanation/explanation_CLS.txt
clusterSize=30
percentage=90

savePath=eraser_movie/layer${layer}/explanation/cluster_Labels_${percentage}%
mkdir -p ${savePath}

saveFile=${savePath}/clusterLabel_layer${layer}.json
echo "Layer${layer}"
python src/generate_explanation_files/generate_all_tokens_explanation.py \
    -i ${clusterPath}/clusters-${clusterSize}.txt \
    -e ${explanation} \
    -s ${saveFile}