#!/bin/bash

scriptDir=src/clustering
inputPath=data/ # path to a sentence file
input=movie_dev_subset.txt #name of the sentence file
dirName=eraser_movie_dev
# put model name or path to a finetuned model for "xxx"
model="best_codebert_model"

# maximum sentence length
sentence_length=512

# analyze latent concepts of layer 12
layer=12

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
maxfreq=15000
delfreq=15000
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
CLUSTERS=300,300,300

python -u ${scriptDir}/extract_data.py --input-file $DATASETPATH --output-path $outputDir

echo "Creating Clusters!"
python -u ${scriptDir}/get_agglomerative_clusters.py \
      --vocab-file $VOCABFILE \
      --point-file $POINTFILE \
      --output-path $RESULTPATH \
      --cluster $CLUSTERS \
      --range 1
echo "DONE!"
