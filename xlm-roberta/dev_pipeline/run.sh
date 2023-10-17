#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --account=def-emilios
#SBATCH --mem=128G 
#SBATCH --mail-type=ALL # 发送哪一种email通知：BEGIN,END,FAIL,ALL
#SBATCH --mail-user=xm863858@dal.ca # 把通知发送到哪一个邮箱


# 1. Extract Representation of input sentence
input=text.in
model=../../glue-eraser-movie-bert-base-cased
sentence_length=300

path=${input}_Result
mkdir $path
cd $path

inputPath=../../../data
scriptDir=../extract_representation

working_file=$input.tok.sent_len #do not change this

# 1.1 Tokenize text with moses tokenizer
perl ${scriptDir}/tokenizer/tokenizer.perl -l en -no-escape < ${inputPath}/$input > $input.tok

# 1.2 Do sentence length filtering and keep sentences max length of 300
python ${scriptDir}/sentence_length.py --text-file $input.tok --length ${sentence_length} --output-file $input.tok.sent_len

# 1.3 Modify the input file to be compatible with the model
python ${scriptDir}/modify_input.py --text-file $input.tok.sent_len --output-file $input.tok.sent_len.modified

# 1.4 Calculate vocabulary size
python ${scriptDir}/frequency_count.py --input-file ${working_file}.modified --output-file ${working_file}.words_freq

working_file=$input.tok.sent_len #do not change this

layers="0 6 11 12"
for layer in $layers
do
echo Layer$layer:
outputDir=layer${layer} #do not change this
mkdir ${outputDir}

# 1.5 Extract layer-wise activations
python ../extract_representation/neurox_extraction.py \
     --model_desc ${model} \
     --input_corpus ${working_file} \
     --output_file ${outputDir}/${working_file}.activations.json \
     --output_type json \
     --decompose_layers \
     --include_special_tokens \
     --filter_layers ${layer} \
     --input_type text

# 1.6. Create a dataset file with word and sentence indexes
python ../extract_representation/create_data_single_layer.py --text-file ${working_file}.modified --activation-file ${outputDir}/${working_file}.activations-layer${layer}.json --output-prefix ${outputDir}/${working_file}-layer${layer}

# 1.7 Filter number of tokens to fit in the memory for clustering. Input file will be from step 4. minfreq sets the minimum frequency. If a word type appears is coming less than minfreq, it will be dropped. if a word comes
minfreq=0
maxfreq=1000000
delfreq=1000000
python ../extract_representation/frequency_filter_data.py --input-file ${outputDir}/${working_file}-layer${layer}-dataset.json --frequency-file ${working_file}.words_freq --sentence-file ${outputDir}/${working_file}-layer${layer}-sentences.json --minimum-frequency $minfreq --maximum-frequency $maxfreq --delete-frequency ${delfreq} --output-file ${outputDir}/${working_file}-layer${layer}


# 2. IG-backpropagation
scriptDir=../IG_backpropagation
inputFile=${input}.tok.sent_len

outDir=IG_attributions
mkdir ${outDir}

saveFile=${outDir}/IG_explanation_layer_${layer}.csv
python ${scriptDir}/ig_for_sequence_classification.py ${inputFile} ${model} $layer ${saveFile}

# 3. Get IG explanation file
scriptDir=../generate_explanation_files
inputDir=IG_attributions
outDir=IG_explanation_files_attribution_mass_50

mkdir ${outDir}

saveFile=${outDir}/explanation_layer_${layer}.txt
python ${scriptDir}/generate_IG_explanation_salient_words.py ${inputDir}/IG_explanation_layer_${layer}.csv ${saveFile} top-k --attribution_mass 0.5

# 4. Match Representations of Salient words
scriptDir=../concept_mapper

dataPath=.
minfreq=0
maxfreq=1000000
delfreq=1000000

savePath=saliency_representation_info #position_representation_info
mkdir $savePath

explanationPath=IG_explanation_files_attribution_mass_50
explanation=${explanationPath}/explanation_layer_$layer.txt
saveFile=$savePath/explanation_words_representation_$layer.csv

python ${scriptDir}/match_representation.py --datasetFile $dataPath/layer$layer/${working_file}-layer${layer}_min_${minfreq}_max_${maxfreq}_del_${delfreq}-dataset.json --explanationFile $explanation --outputFile $saveFile

# 5. Concept Mapper
fileDir=saliency_representation_info 
classifierDir=../../classifier_mapping/result/model

python ${scriptDir}/logistic_regression.py \
  --test_file_path ${fileDir}/explanation_words_representation_${layer}.csv \
  --layer ${layer} \
  --save_path ./latent_concepts/saliency_prediction/ \
  --do_predict \
  --classifier_file_path $classifierDir/layer_${layer}_classifier.pkl \
  --load_classifier_from_local
done
