#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64
#SBATCH --job-name="run_model"
#SBATCH --mail-user=vedant29@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="logs/slurm-%j.out"

scriptDir=src/IG_backpropagation
inputFile=eraser_movie/movie_train.txt.tok.sent_len
model="/lustre/hdd/LAS/jannesar-lab/vedant29/CodeXGLUE/Code-Code/Defect-detection/code/saved_models/checkpoint-best-acc"

outDir=eraser_movie/IG_attributions
mkdir -p ${outDir}

# Loop over layers 0 to 12
for layer in {0..12}; do
    saveFile=${outDir}/IG_explanation_layer_${layer}.csv

    echo "Running for layer ${layer}..."
    python ${scriptDir}/ig_for_sequence_classification.py \
        --input_file ${inputFile} \
        --model ${model} \
        --layer ${layer} \
        --save_file ${saveFile} \
        --model_type roberta \
        --pool_all_tokens
done
