#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --account=def-emilios
#SBATCH --mem=128G
#SBATCH --mail-type=ALL # 发送哪一种email通知：BEGIN,END,FAIL,ALL
#SBATCH --mail-user=xm863858@dal.ca # 把通知发送到哪一个邮箱

scriptDir=scripts
inputPath=../../data # path to a sentence file
input=movie_train.txt #name of the sentence file

mkdir "eraser_movie"

# maximum sentence length
sentence_length=300
# analyze latent concepts of layer 12

working_file=$input.tok.sent_len #do not change this

#1. Tokenize text with moses tokenizer
perl ${scriptDir}/tokenizer/tokenizer.perl -l en -no-escape < ${inputPath}/$input > $input.tok

#2. Do sentence length filtering and keep sentences max length of 300
python ${scriptDir}/sentence_length.py --text-file $input.tok --length ${sentence_length} --output-file $input.tok.sent_len

#3. Modify the input file to be compatible with the model
python ${scriptDir}/modify_input.py --text-file $input.tok.sent_len --sentence-tag "<s>" --output-file $input.tok.sent_len.modified

#4. Calculate vocabulary size
python ${scriptDir}/frequency_count.py --input-file ${working_file}.modified --output-file ${working_file}.words_freq

