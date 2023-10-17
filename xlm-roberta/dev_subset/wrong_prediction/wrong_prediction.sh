#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --account=def-hsajjad
#SBATCH --mem=128G
#SBATCH --mail-type=ALL # 发送哪一种email通知：BEGIN,END,FAIL,ALL
#SBATCH --mail-user=xm863858@dal.ca # 把通知发送到哪一个邮箱

scriptDir=scripts
savePath=wrong_predictions_info_saliency
#explanation=dev/explanation_words.txt

predictionFile='../generate_explanation_files/predicted_labels(original).csv'
goldFile='../../../data/movie_dev_subset.json'

mkdir $savePath
layer=12
explanation=../IG_explanation_files_attribution_mass_50/explanation_layer_${layer}.txt
saveFile=${savePath}/wrong_predictions_info_$layer.csv

python get_wrong_predictions.py --gold_label_file $goldFile --predicted_file $predictionFile --explanation_file $explanation --output_file $saveFile
