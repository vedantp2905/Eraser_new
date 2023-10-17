#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --account=def-hsajjad
#SBATCH --mem=512G 
#SBATCH --mail-type=ALL # 发送哪一种email通知：BEGIN,END,FAIL,ALL
#SBATCH --mail-user=xm863858@dal.ca # 把通知发送到哪一个邮箱

scriptsDir=scripts
fileDir=split_dataset #split_dataset_CLS
savePath=result #result_CLS

for layer in {0..12}
do
echo $layer
python ${scriptsDir}/logistic_regression.py --train_file_path ${fileDir}/train/train_df_${layer}.csv --validate_file_path ${fileDir}/validation/validation_df_${layer}.csv --layer ${layer} --save_path ${savePath} --do_train --do_validate
done

# layer=12
# python logistic_regression.py \
#   --train_file_path ./clusters_csv_train/clusters-map${layer}.csv \
#   --test_file_path ./clusters_csv_test/clusters-map${layer}.csv \
#   --layer ${layer} \
#   --save_path ./result2 \
#   --do_train \
#   --do_predict \
#   --classifier_file_path ./result/model/layer_${layer}_classifier.pkl \
#   --load_classifier_from_local