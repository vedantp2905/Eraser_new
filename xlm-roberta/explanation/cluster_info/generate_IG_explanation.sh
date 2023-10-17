#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --account=def-hsajjad
#SBATCH --mem=512G 
#SBATCH --mail-type=ALL # 发送哪一种email通知：BEGIN,END,FAIL,ALL
#SBATCH --mail-user=xm863858@dal.ca # 把通知发送到哪一个邮箱

scriptDir=scripts
inputDir=../../IG_attributions
outDir=../IG_explanation_files_mass_50
# outDir=../IG_explanation_top

mkdir ${outDir}

for layer in {0..12}
do
echo ${inputDir}/IG_explanation_layer_${layer}.csv
saveFile=${outDir}/explanation_layer_${layer}.txt
# python ${scriptDir}/generate_IG_explanation.py ${inputDir}/IG_explanation_layer_${layer}.csv ${saveFile}
python ${scriptDir}/generate_IG_explanation_salient_words.py ${inputDir}/IG_explanation_layer_${layer}.csv ${saveFile} top-k --attribution_mass 0.5
# python ${scriptDir}/generate_IG_explanation_salient_words.py ${inputDir}/IG_explanation_layer_${layer}.csv ${saveFile} top-1
done

# outDir=../IG_explanation_files

# mkdir ${outDir}

# for layer in {0..12}
# do
# echo ${inputDir}/IG_explanation_layer_${layer}.csv
# saveFile=${outDir}/explanation_layer_${layer}.txt
# # python ${scriptDir}/generate_IG_explanation.py ${inputDir}/IG_explanation_layer_${layer}.csv ${saveFile}
# python ${scriptDir}/generate_IG_explanation.py ${inputDir}/IG_explanation_layer_${layer}.csv ${saveFile}
# done
