#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --account=def-hsajjad
#SBATCH --mem=128G
#SBATCH --mail-type=ALL # 发送哪一种email通知：BEGIN,END,FAIL,ALL
#SBATCH --mail-user=xm863858@dal.ca # 把通知发送到哪一个邮箱

scriptDir=scripts
layer=12
wrong_prediction_file=wrong_predictions_info_saliency/wrong_predictions_info_$layer.csv
predicted_concepts_file=../concept_mapper/latent_concepts/saliency_prediction/predictions_layer_$layer.csv

outputPath=wrong_prediction_saliency_predictions/
mkdir $outputPath

output_file=$outputPath/explanation_latent_concept_$layer.csv

python match_with_predicted_concepts.py --wrong_prediction_file $wrong_prediction_file --predicted_concepts_file $predicted_concepts_file --output_file $output_file
