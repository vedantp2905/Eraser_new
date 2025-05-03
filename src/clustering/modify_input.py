import argparse
import pandas as pd


# read the input.in.tok.sent_len file
def get_dataset(dataset_name, text_file):
    if dataset_name == "stsb":
        columns = ["sentence1", "sentence2"]
        dataset = pd.read_csv(text_file, delimiter='\t', header=None, names=columns)
    else:
        with open(text_file) as f:
            dataset = f.readlines()
    return dataset


# add [CLS] at the beginning of each sentence and [SEP] at the end of each sentence
def add_special_tokens(dataset, sentence_tag, dataset_name):
    format_data = []

    start_tag = "[CLS]"
    end_tag = "[SEP]"

    if sentence_tag == "<s>":
        start_tag = "<s>"
        end_tag = "</s>"
    
    for i in range(len(dataset)):
        if dataset_name == "stsb":
            format_data.append(start_tag + " " + dataset["sentence1"][i] + " " + end_tag + " " + dataset["sentence2"][i] + " " + end_tag)
        else:
            format_data.append(start_tag + " " + dataset[i][:-1] + " " + end_tag)

    return format_data


# save the modified dataset to a file
# each line is a sentence
def save_dataset(dataset):
    with open(args.output_file, "w") as f:
        f.write("\n".join(dataset))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--text-file', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    parser.add_argument('--sentence-tag', type=str, default="[CLS]", help="sentence tag used in the tokenizer ([CLS] or <s>)")
    parser.add_argument('--dataset-name', type=str, help="name of the dataset")

    args = parser.parse_args()

    dataset = get_dataset(args.dataset_name, args.text_file)
    dataset = add_special_tokens(dataset, args.sentence_tag, args.dataset_name)
    save_dataset(dataset)
