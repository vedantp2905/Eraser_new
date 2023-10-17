#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --account=def-emilios
#SBATCH --mem=128G
#SBATCH --mail-type=ALL # 发送哪一种email通知：BEGIN,END,FAIL,ALL
#SBATCH --mail-user=xm863858@dal.ca # 把通知发送到哪一个邮箱

#layer=12
#python get_prediction_stats.py \
#  --layer ${layer} \
#  --file_path ./result/validate_predictions/

fileDir=result_CLS/validate_predictions/
scriptsDir=scripts

# python ${scriptsDir}/get_prediction_stats.py \
#   --all_layer_stats \
#   --file_path ${fileDir}

python ${scriptsDir}/get_prediction_stats.py \
  --layer 4,5,6,7,8,9,10,11,12 \
  --file_path ${fileDir}