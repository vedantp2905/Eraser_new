#!/bin/bash

scriptDir=../../../src/clustering
inputPath=../../../data/ # path to a sentence file
input=movie_train.txt #name of the sentence file

mkdir "eraser_movie"

# maximum sentence length
sentence_length=300

working_file=$input.tok.sent_len #do not change this

#1. Tokenize text with moses tokenizer
perl ${scriptDir}/tokenizer/tokenizer.perl -l en -no-escape < ${inputPath}/$input > $input.tok

#2. Do sentence length filtering and keep sentences max length of 300
python ${scriptDir}/sentence_length.py --text-file $input.tok --length ${sentence_length} --output-file $input.tok.sent_len

#3. Modify the input file to be compatible with the model
python ${scriptDir}/modify_input.py --text-file $input.tok.sent_len --output-file $input.tok.sent_len.modified

#4. Calculate vocabulary size
python ${scriptDir}/frequency_count.py --input-file ${working_file}.modified --output-file ${working_file}.words_freq

