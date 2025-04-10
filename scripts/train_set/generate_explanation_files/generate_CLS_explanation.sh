#!/bin/bash

scriptDir=src/generate_explanation_files
model="C:/Users/91917/Desktop/Research/eraser_movie_latentConcept/best_codebert_model" # add the path to the model to "xxx"
inputFile=../clustering/movie_train.txt.tok

saveDir=.   

layer=12
python ${scriptDir}/generate_CLS_explanation.py --dataset-name-or-path ${inputFile} --model-name ${model} --tokenizer-name ${model} --save-dir ${saveDir}



