#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --account=def-hsajjad
#SBATCH --mem=128G 
#SBATCH --mail-type=ALL # 发送哪一种email通知：BEGIN,END,FAIL,ALL
#SBATCH --mail-user=xm863858@dal.ca # 把通知发送到哪一个邮箱

scriptDir=scripts
input=movie_dev_subset.txt
working_file=$input.tok.sent_len

dataPath=../extract_representation/dev_subset
minfreq=0
maxfreq=1000000
delfreq=1000000

savePath=.

explanation=explanation_CLS.txt
saveFile=$savePath/explanation_words.txt

layer=12
python ${scriptDir}/generate_all_tokens_explanation_dev_set.py -i $dataPath/layer$layer/${working_file}-layer${layer}_min_${minfreq}_max_${maxfreq}_del_${delfreq}-dataset.json -e $explanation -s $saveFile

