""" Generate the explanation file for [CLS] tokens.

This scripts will get the predicted label for each sentence in the dataset, and then use the predicted label,
sentence_idx and word_idx to generate the explanation file for CLS tokens.
"""

import argparse

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import os
import json


def get_dataset(file_name):
    with open(file_name, 'r') as f:
        data = f.readlines()

    data = [line.strip() for line in data]

    return data


def get_hidden_states_inputs(model, tokenizer, sentence, device):
    """
    Input the sentences into the model to get the hidden states and predicted label of the model.

    Parameters
    ----------
    model: transformers.modeling_utils.PreTrainedModel
        The pre-trained model to be used.

    tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase
        The tokenizer to be used.

    dataset: datasets.arrow_dataset.Dataset
        The dataset to be tokenized.
    """

    input_text = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**input_text)

    predictions = F.softmax(outputs[0], dim=-1)
    labels = torch.argmax(predictions, axis=1)  # 0: Negative, 1: Positive
    labels = labels.tolist()

    return input_text, labels


def save_prediction(labels, sentence, output_path):
    path = output_path + '/predicted.csv'

    df = pd.DataFrame({'sentence': sentence, 'labels': labels})
    df.to_csv(path, sep='\t', index=False)


def generate_explanation(ids, labels, save_path):
    """
    Generate and save the explanation for the [CLS] tokens.

    Parameters
    ----------
    ids: torch.Tensor
        The token ids of the sentences.

    labels: List
        The predicted labels of the sentences.

    save_path: str
        The path to save the json file.

    Returns
    -------
    layer_cls_info: list
        The [CLS] token representation and info.
    """
    predictions = []

    for j in range(len(ids)):  # len(ids) => # of sentences
        print(j, ids[j])
        predicted_result = labels[j]

        # prediction class(label) ||| position_id ||| sentence_id
        predictions.append(str(predicted_result) + " " + str(-1) + " " + str(j))

    # check the directory exists or not
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    path = save_path + '/explanation_CLS.txt'
    # save the explanation in a txt file
    with open(path, "w") as txt_file:
        for line in predictions:
            txt_file.write(line + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name-or-path',
                        type=str,
                        default='./saved_sst2',
                        help='The name or path of the dataset to be loaded.')

    # parser.add_argument('--task-name',
    #                     type=str,
    #                     help="name of the task dataset")

    parser.add_argument('--model-name',
                        type=str,
                        default='bert-base-cased',
                        help='The name or path of the pre-trained model to be used.')

    parser.add_argument('--tokenizer-name',
                        type=str,
                        default='bert-base-cased',
                        help='The name or path of the tokenizer to be used.')

    parser.add_argument('--save-dir',
                        type=str,
                        default='CLS_tokens/',
                        help='The directory to save the extracted [CLS] token representation and info.')

    args = parser.parse_args()


    sentence = get_dataset(args.dataset_name_or_path)
    # dataset = get_dataset(args.dataset_name_or_path, args.task_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name).to(device)

    print("Finishing loading the dataset, model and tokenizer.")

    inputs, labels = get_hidden_states_inputs(model, tokenizer, sentence, device)
    print("Finishing getting the hidden states and predicted labels.")

    # save_prediction(labels, dataset['sentence1'], dataset['sentence2'], args.save_dir)

    save_prediction(labels, sentence, args.save_dir)

    ids = inputs['input_ids']

    generate_explanation(ids, labels, args.save_dir)
    print("Finishing generating the explanation for [CLS] tokens.")


if __name__ == '__main__':
    main()

