#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=64G
#SBATCH --job-name="run_model"
#SBATCH --mail-user=vedant29@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="logs/slurm-%j.out"

source /lustre/hdd/LAS/jannesar-lab/vedant29/miniconda3/bin/activate neurox_pip
# Check if model parameter is provided and validate it
model=${1:-""}
if [[ -z "$model" ]]; then
    echo "Running without model specification..."
    results_dir="results"
else
    echo "Using model: $model"
    results_dir="results/${model}"
fi

# Check for required directories and files
if [ ! -d "data" ]; then
    echo "Error: data directory not found. Creating it..."
    mkdir -p data
fi

if [ ! -f "data/movie_train.txt" ]; then
    echo "Error: data/movie_train.txt not found"
    exit 1
fi

if [ ! -f "data/movie_dev.txt" ]; then
    echo "Error: data/movie_dev.txt not found"
    exit 1
fi

# Create required directories
mkdir -p src/clustering/tokenizer
mkdir -p ${results_dir}/train/clustering
mkdir -p ${results_dir}/dev/clustering

echo "Starting Training Phase..."

# Convert line endings and make scripts executable
find scripts/ -name "*.sh" -type f -exec dos2unix {} \;
find scripts/ -name "*.sh" -type f -exec chmod +x {} \;

# 1. Extract Latent Concepts
echo "1. Extracting Latent Concepts..."
bash scripts/train_set/clustering/clustering_base_work.sh "$model"
if [ $? -ne 0 ]; then echo "Error in clustering_base_work.sh"; exit 1; fi
bash scripts/train_set/clustering/clustering_conceptX.sh "$model"
if [ $? -ne 0 ]; then echo "Error in clustering_conceptX.sh"; exit 1; fi

# 2. Train ConceptMapper
echo "2. Training ConceptMapper..."
bash scripts/train_set/classifier_mapping/generate_csv_file.sh "$model"
if [ $? -ne 0 ]; then echo "Error in generate_csv_file.sh"; exit 1; fi
bash scripts/train_set/classifier_mapping/split_dataset.sh "$model"
if [ $? -ne 0 ]; then echo "Error in split_dataset.sh"; exit 1; fi
bash scripts/train_set/classifier_mapping/logistic_regression.sh "$model"
if [ $? -ne 0 ]; then echo "Error in logistic_regression.sh"; exit 1; fi
bash scripts/train_set/classifier_mapping/get_prediction_stat.sh "$model"
if [ $? -ne 0 ]; then echo "Error in get_prediction_stat.sh"; exit 1; fi

# 3. Discover Salient Tokens
echo "3. Discovering Salient Tokens..."
bash scripts/train_set/IG_backpropagation/ig.sh "$model"
if [ $? -ne 0 ]; then echo "Error in ig.sh"; exit 1; fi

# 4. Get Predictions for Training Set
echo "4. Getting Training Set Predictions..."
bash scripts/train_set/generate_explanation_files/generate_CLS_explanation.sh "$model"
if [ $? -ne 0 ]; then echo "Error in generate_CLS_explanation.sh"; exit 1; fi
bash scripts/train_set/generate_explanation_files/generate_IG_explanation.sh "$model"
if [ $? -ne 0 ]; then echo "Error in generate_IG_explanation.sh"; exit 1; fi
bash scripts/train_set/generate_explanation_files/generate_all_tokens_explanation.sh "$model"
if [ $? -ne 0 ]; then echo "Error in generate_all_tokens_explanation.sh"; exit 1; fi


echo "Starting Inference Phase..."

# 1. Extract Latent Concepts for Dev Set
echo "1. Extracting Latent Concepts for Dev Set..."
bash scripts/dev_set/clustering/clustering_base_work.sh "$model"
if [ $? -ne 0 ]; then echo "Error in dev clustering_base_work.sh"; exit 1; fi
bash scripts/dev_set/clustering/clustering_conceptX.sh "$model"
if [ $? -ne 0 ]; then echo "Error in dev clustering_conceptX.sh"; exit 1; fi

# 2. Discover Salient Tokens for Dev Set
echo "2. Discovering Salient Tokens for Dev Set..."
bash scripts/dev_set/IG_backpropagation/ig.sh "$model"
if [ $? -ne 0 ]; then echo "Error in dev ig.sh"; exit 1; fi

# 3. Get Predictions for Dev Set
echo "3. Getting Dev Set Predictions..."
bash scripts/dev_set/generate_explanation_files/generate_CLS_explanation.sh "$model"
if [ $? -ne 0 ]; then echo "Error in dev generate_CLS_explanation.sh"; exit 1; fi
bash scripts/dev_set/generate_explanation_files/generate_IG_explanation.sh "$model"
if [ $? -ne 0 ]; then echo "Error in dev generate_IG_explanation.sh"; exit 1; fi
bash scripts/dev_set/generate_explanation_files/generate_all_tokens_explanation.sh "$model"
if [ $? -ne 0 ]; then echo "Error in dev generate_all_tokens_explanation.sh"; exit 1; fi

# 4. Map Tokens to Latent Concepts
echo "4. Mapping Tokens to Latent Concepts..."
bash scripts/dev_set/concept_mapper/match_representation.sh "$model"
if [ $? -ne 0 ]; then echo "Error in match_representation.sh"; exit 1; fi
bash scripts/dev_set/concept_mapper/logistic_regression.sh "$model"
if [ $? -ne 0 ]; then echo "Error in dev logistic_regression.sh"; exit 1; fi

echo "All processes completed!"
