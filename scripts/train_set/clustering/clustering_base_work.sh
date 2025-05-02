#!/bin/bash

scriptDir=src/clustering
inputPath=data/ # path to a sentence file
input=movie_train.txt #name of the sentence file
dirName="eraser_movie"
mkdir $dirName

# Accept sentence length argument
sentence_length=${1}

working_file=$input.tok.sent_len #do not change this

#1. Tokenize text with moses tokenizer
perl ${scriptDir}/tokenizer/tokenizer.perl -l en -no-escape < ${inputPath}/$input > $dirName/$input.tok

#2. Do sentence length filtering and keep sentences max length of 512
python ${scriptDir}/sentence_length.py --text-file $dirName/$input.tok --length ${sentence_length} --output-file $dirName/$input.tok.sent_len

#3. Modify the input file to be compatible with the model
python ${scriptDir}/modify_input.py --text-file $dirName/$input.tok.sent_len --output-file $dirName/$input.tok.sent_len.modified

#4. Calculate vocabulary size
python ${scriptDir}/frequency_count.py --input-file $dirName/$input.tok.sent_len.modified --output-file $dirName/${working_file}.words_freq

