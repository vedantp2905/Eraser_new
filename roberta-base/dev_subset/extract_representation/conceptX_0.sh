#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --account=def-emilios
#SBATCH --mem=512G 
#SBATCH --mail-type=ALL # 发送哪一种email通知：BEGIN,END,FAIL,ALL
#SBATCH --mail-user=xm863858@dal.ca # 把通知发送到哪一个邮箱

scriptDir=scripts
inputPath=../../../data # path to a sentence file
input=movie_dev_subset.txt #name of the sentence file
model=../../glue-eraser-movie-roberta-base-cased

# maximum sentence length
sentence_length=300
# analyze latent concepts of layer 12
layer=0

outputDir=dev_subset/layer${layer} #do not change this
mkdir ${outputDir}

working_file=$input.tok.sent_len #do not change this

# source activate neurox_pip
# 3. Extract layer-wise activations
python ${scriptDir}/neurox_extraction.py \
     --model_desc ${model} \
     --input_corpus ${working_file} \
     --output_file ${outputDir}/${working_file}.activations.json \
     --output_type json \
     --decompose_layers \
     --include_special_tokens \
     --filter_layers ${layer} \
     --input_type text

#4. Create a dataset file with word and sentence indexes
python ${scriptDir}/create_data_single_layer.py --text-file ${working_file}.modified --activation-file ${outputDir}/${working_file}.activations-layer${layer}.json --sentence-tag "<s>" --output-prefix ${outputDir}/${working_file}-layer${layer}

#6. Filter number of tokens to fit in the memory for clustering. Input file will be from step 4. minfreq sets the minimum frequency. If a word type appears is coming less than minfreq, it will be dropped. if a word comes
minfreq=0
maxfreq=1000000
delfreq=1000000
python ${scriptDir}/frequency_filter_data.py --input-file ${outputDir}/${working_file}-layer${layer}-dataset.json --frequency-file ${working_file}.words_freq --sentence-file ${outputDir}/${working_file}-layer${layer}-sentences.json --minimum-frequency $minfreq --maximum-frequency $maxfreq --delete-frequency ${delfreq} --sentence-tag "<s>" --output-file ${outputDir}/${working_file}-layer${layer}
