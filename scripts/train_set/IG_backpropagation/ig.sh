#!/bin/bash

#SBATCH --time=24:00:00          # Time limit (24 hours)
#SBATCH --nodes=1                # Request 1 node
#SBATCH --ntasks-per-node=8      # Request 8 tasks per node (adjust as needed)
#SBATCH --gres=gpu:a100:1       # Request 2 A100 GPUs (adjust as needed)
#SBATCH --mem=128GB              # Request 128 GB of memory (adjust if needed)
#SBATCH --job-name="run_model"   # Job name
#SBATCH --mail-user=vedant29@iastate.edu  # Email for notifications
#SBATCH --mail-type=BEGIN        # Notify when the job starts
#SBATCH --mail-type=END          # Notify when the job finishes
#SBATCH --mail-type=FAIL         # Notify if the job fails
#SBATCH --output="logs/slurm-%j.out"  # Output file for job logs


scriptDir=src/IG_backpropagation
inputFile=eraser_movie/movie_train.txt.tok.sent_len
model="/lustre/hdd/LAS/jannesar-lab/vedant29/finetuned_models/codebert-pos-lang-classification/best_model"

outDir=eraser_movie/IG_attributions
mkdir -p ${outDir}

# Loop over layers 0 to 12
for layer in {6..6}; do
    saveFile=${outDir}/IG_explanation_layer_${layer}.csv

    echo "Running for layer ${layer}..."
    python ${scriptDir}/ig_for_sequence_classification.py \
        --input_file ${inputFile} \
        --model ${model} \
        --layer ${layer} \
        --save_file ${saveFile} \
        --model_type roberta 
done
