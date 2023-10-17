""" Generate the explanation file for [CLS] tokens.

This scripts will get the predicted label for each sentence in the dataset, and then use the predicted label,
sentence_idx and word_idx to generate the explanation file for CLS tokens.
"""

import argparse

import pandas as pd
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import os
import json


def get_dataset(file_name, task_name):
    try:
        dataset = load_from_disk(file_name)
        dataset = dataset['train']

        # If the dataset is loaded from the datasets library, using the following code
        # from datasets import load_dataset
        # dataset = load_dataset('glue', 'sst2')
    except:
        # read the dataset as json file from the local directory
        dataset = json.load(open(file_name, "r"))

        # get the sentences and labels from the dataset
        if task_name == 'stsb':
            sentence1 = []
            sentence2 = []
        else:
            sentences = []

        label = []

        for i in range(len(dataset)):
            if task_name == 'stsb':
                sentence1.append(dataset[i]['sentence1'])
                sentence2.append(dataset[i]['sentence2'])
            else:
                sentences.append(dataset[i]['sentence'])

            label.append(dataset[i]['label'])

        # get the sentences and labels from the dataset
        if task_name == 'stsb':
            dataset = {"sentence1": sentence1, "sentence2": sentence2, "label": label}
        else:
            dataset = {"sentence": sentences, "label": label}

    return dataset


def get_hidden_states_inputs(model, tokenizer, dataset, task_name, device):
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
    if task_name == 'stsb':
        input_text = tokenizer(dataset['sentence1'], dataset['sentence2'], padding=True, truncation=True,
                               return_tensors="pt").to(device)

        with torch.no_grad():
            output = model(**input_text)

        logits = output.logits
        predicted_scores = logits.squeeze().tolist()

        # convert the score to a value between 0 and 5
        # Define the minimum and maximum values of the predicted scores
        min_pred = min(predicted_scores)
        max_pred = max(predicted_scores)

        # Define the desired range of the labels
        min_label = 0
        max_label = 5

        # Scale the predicted scores to the range of 0 to 5
        labels = [(score - min_pred) / (max_pred - min_pred) * (max_label - min_label) + min_label for score in
                  predicted_scores]

    else:
        input_text = tokenizer(dataset['sentence'], padding=True, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**input_text)

        predictions = F.softmax(outputs[0], dim=-1)
        labels = torch.argmax(predictions, axis=1)  # 0: Negative, 1: Positive
        labels = labels.tolist()

    return input_text, labels


def save_prediction(labels, dataset, task_name, output_path):
    path = output_path + '/predicted_labels(original).csv'
    if task_name == 'stsb':
        df = pd.DataFrame({'sentence1': dataset['sentence1'], 'sentence2': dataset['sentence2'], 'labels': labels})
    else:
        df = pd.DataFrame({'sentence': dataset['sentence'], 'labels': labels})
    df.to_csv(path, sep='\t', index=False)


def convert_label(label):
    bucket_label = []
    for val in label:
        if val < 2.5:
            val = 0
        else:
            val = 1
        bucket_label.append(val)
    return bucket_label


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

    parser.add_argument('--task-name',
                        type=str,
                        help="name of the task dataset")

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

    dataset = get_dataset(args.dataset_name_or_path, args.task_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name).to(device)

    print("Finishing loading the dataset, model and tokenizer.")

    inputs, labels = get_hidden_states_inputs(model, tokenizer, dataset, args.task_name, device)
    print("Finishing getting the hidden states and predicted labels.")

    # save_prediction(labels, dataset['sentence1'], dataset['sentence2'], args.save_dir)

    save_prediction(labels, dataset, args.task_name, args.save_dir)

    if args.task_name == 'stsb':
        labels = convert_label(labels)

    ids = inputs['input_ids']

    generate_explanation(ids, labels, args.save_dir)
    print("Finishing generating the explanation for [CLS] tokens.")


if __name__ == '__main__':
    main()