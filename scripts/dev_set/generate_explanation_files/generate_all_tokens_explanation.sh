#!/bin/bash

# Accept layer argument
layer=${1}

clusterPath=eraser_movie_dev/layer${layer}/results
explanation=eraser_movie_dev/layer${layer}/explanation/explanation_CLS.txt
clusterSize=${2}
percentage=90

savePath=eraser_movie_dev/layer${layer}/explanation/cluster_Labels_${percentage}%
mkdir -p ${savePath}

saveFile=${savePath}/clusterLabel_layer${layer}.json
echo "Layer${layer}"
python src/generate_explanation_files/generate_all_tokens_explanation.py \
    -i ${clusterPath}/clusters-${clusterSize}.txt \
    -e ${explanation} \
    -s ${saveFile}