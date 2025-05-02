#!/bin/bash

scriptDir=src/clustering
inputPath=data/ # path to a sentence file
input=movie_train.txt #name of the sentence file
dirName=eraser_movie
# put model name or path to a finetuned model for "xxx"

# Accept model and layer arguments
layer=${1}
model=${2}
sentence_length=${3}
cluster_num=${4}

outputDir=${dirName}/layer${layer} #do not change this
mkdir -p ${outputDir}

# Reference the files in dirName directory created by clustering_base_work.sh
working_file=${dirName}/${input}.tok.sent_len

# Extract layer-wise activations
python ${scriptDir}/neurox_extraction.py \
      --model_desc ${model} \
      --input_corpus ${working_file} \
      --output_file ${outputDir}/$(basename ${working_file}).activations.json \
      --output_type json \
      --decompose_layers \
      --include_special_tokens \
      --filter_layers ${layer} \
      --input_type text

# Create a dataset file with word and sentence indexes
python ${scriptDir}/create_data_single_layer.py \
      --text-file ${working_file}.modified \
      --activation-file ${outputDir}/$(basename ${working_file}).activations-layer${layer}.json \
      --output-prefix ${outputDir}/$(basename ${working_file})-layer${layer}

# Filter number of tokens
minfreq=0
maxfreq=10000000
delfreq=10000000
python ${scriptDir}/frequency_filter_data.py \
      --input-file ${outputDir}/$(basename ${working_file})-layer${layer}-dataset.json \
      --frequency-file ${working_file}.words_freq \
      --sentence-file ${outputDir}/$(basename ${working_file})-layer${layer}-sentences.json \
      --minimum-frequency $minfreq \
      --maximum-frequency $maxfreq \
      --delete-frequency ${delfreq} \
      --output-file ${outputDir}/$(basename ${working_file})-layer${layer}

# Run clustering
mkdir -p ${outputDir}/results
DATASETPATH=${outputDir}/$(basename ${working_file})-layer${layer}_min_${minfreq}_max_${maxfreq}_del_${delfreq}-dataset.json
VOCABFILE=${outputDir}/processed-vocab.npy
POINTFILE=${outputDir}/processed-point.npy
RESULTPATH=${outputDir}/results
CLUSTERS=${cluster_num},${cluster_num},${cluster_num}

python -u ${scriptDir}/extract_data.py --input-file $DATASETPATH --output-path $outputDir

echo "Creating Clusters!"
python src/clustering/get_agglomerative_clusters.py \
    --vocab-file eraser_movie/layer${layer}/processed-vocab.npy \
    --point-file eraser_movie/layer${layer}/processed-point.npy \
    --output-path eraser_movie/layer${layer}/results \
    --cluster ${cluster_num} \
    --batch-size 1024
    
echo "DONE!"
