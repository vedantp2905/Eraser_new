#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64
#SBATCH --job-name="run_model"
#SBATCH --mail-user=vedant29@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="logs/slurm-%j.out"

# Initialize Conda
eval "$(conda shell.bash hook)"

# Activate the environment
conda activate neurox_pip

# Create required directories
mkdir -p logs

# Convert line endings and make scripts executable
find scripts/ -name "*.sh" -type f -exec dos2unix {} \;
find scripts/ -name "*.sh" -type f -exec chmod +x {} \;

echo "Starting Training Phase..."

# 1. Extract Latent Concepts
echo "1. Extracting Latent Concepts..."
bash scripts/train_set/clustering/clustering_base_work.sh
if [ $? -ne 0 ]; then echo "Error in clustering_base_work.sh"; exit 1; fi

bash scripts/train_set/clustering/clustering_conceptX.sh
if [ $? -ne 0 ]; then echo "Error in clustering_conceptX.sh"; exit 1; fi

# 2. Train ConceptMapper
echo "2. Training ConceptMapper..."
bash scripts/train_set/classifier_mapping/generate_csv_file.sh
if [ $? -ne 0 ]; then echo "Error in generate_csv_file.sh"; exit 1; fi

bash scripts/train_set/classifier_mapping/split_dataset.sh
if [ $? -ne 0 ]; then echo "Error in split_dataset.sh"; exit 1; fi

bash scripts/train_set/classifier_mapping/logistic_regression.sh
if [ $? -ne 0 ]; then echo "Error in logistic_regression.sh"; exit 1; fi

bash scripts/train_set/classifier_mapping/get_prediction_stat.sh
if [ $? -ne 0 ]; then echo "Error in get_prediction_stat.sh"; exit 1; fi

# 3. Discover Salient Tokens
echo "3. Discovering Salient Tokens..."
bash scripts/train_set/IG_backpropagation/ig.sh
if [ $? -ne 0 ]; then echo "Error in ig.sh"; exit 1; fi

# 4. Get Predictions
echo "4. Getting Predictions..."
bash scripts/train_set/generate_explanation_files/generate_CLS_explanation.sh
if [ $? -ne 0 ]; then echo "Error in generate_CLS_explanation.sh"; exit 1; fi

bash scripts/train_set/generate_explanation_files/generate_all_tokens_explanation.sh
if [ $? -ne 0 ]; then echo "Error in generate_all_tokens_explanation.sh"; exit 1; fi

bash scripts/train_set/generate_explanation_files/generate_IG_explanation.sh
if [ $? -ne 0 ]; then echo "Error in generate_IG_explanation.sh"; exit 1; fi

echo "Starting Inference Phase..."

# 1. Extract Latent Concepts
echo "1. Extracting Latent Concepts..."
bash scripts/dev_set/clustering/clustering_base_work.sh
if [ $? -ne 0 ]; then echo "Error in dev clustering_base_work.sh"; exit 1; fi

bash scripts/dev_set/clustering/clustering_conceptX.sh
if [ $? -ne 0 ]; then echo "Error in dev clustering_conceptX.sh"; exit 1; fi

# 2. Discover Salient Tokens
echo "2. Discovering Salient Tokens..."
bash scripts/dev_set/IG_backpropagation/ig.sh
if [ $? -ne 0 ]; then echo "Error in dev ig.sh"; exit 1; fi

# 3. Get Predictions
echo "3. Getting Predictions..."
bash scripts/dev_set/generate_explanation_files/generate_CLS_explanation.sh
if [ $? -ne 0 ]; then echo "Error in dev generate_CLS_explanation.sh"; exit 1; fi

bash scripts/dev_set/generate_explanation_files/generate_all_tokens_explanation.sh
if [ $? -ne 0 ]; then echo "Error in dev generate_all_tokens_explanation.sh"; exit 1; fi

bash scripts/dev_set/generate_explanation_files/generate_IG_explanation.sh
if [ $? -ne 0 ]; then echo "Error in dev generate_IG_explanation.sh"; exit 1; fi

# 4. Map Tokens to Latent Concepts
echo "4. Mapping Tokens to Latent Concepts..."
bash scripts/dev_set/concept_mapper/match_representation.sh
if [ $? -ne 0 ]; then echo "Error in match_representation.sh"; exit 1; fi

bash scripts/dev_set/concept_mapper/logistic_regression.sh
if [ $? -ne 0 ]; then echo "Error in dev logistic_regression.sh"; exit 1; fi

echo "All processes completed!"
