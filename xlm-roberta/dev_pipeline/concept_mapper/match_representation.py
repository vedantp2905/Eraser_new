import json
from argparse import ArgumentParser

import pandas as pd


def fileRead(fname, flag):
    lines = []

    if (flag == 0):
        with open(fname, "r") as f:
            data = json.load(f)

        for line in data:
            line = line.rstrip('\r\n')
            lines.append(line)
    else:
        print("Reading file: " + fname)
        with open(fname) as f:
            for line in f:
                line = line.rstrip('\r\n')
                lines.append(line)

    f.close()
    return lines


# read the json file [text.in.tok.sent_len_min_0_max_1000000_del_1000000-dataset.json] (token & token representation)
def read_dataset_file(fname):
    dataset_map = {}
    with open(fname, 'r') as json_file:
        json_data = json.load(json_file)

        for value in json_data:
            token = value[0].split("|||")[0]
            sentence_idx = value[0].split("|||")[2]
            word_idx = value[0].split("|||")[3]
            token_rep = value[1]

            dataset_map[word_idx + '_' + sentence_idx] = (token, token_rep)

        return dataset_map


def match_to_representation(explanationFile, dataset_map):
    all_sentence_idx = []
    all_word_idx = []
    all_tokens = []
    all_token_rep = []
    all_labels = []

    for i, w in enumerate(explanationFile):
        try:
            label = w.split(" ")[0]
            sentence_idx = w.split(" ")[2]
            word_idx = w.split(" ")[1]
            all_sentence_idx.append(sentence_idx)
            all_word_idx.append(word_idx)
            all_labels.append(label)

            token, token_rep = dataset_map[word_idx + '_' + sentence_idx]
            all_tokens.append(token)
            all_token_rep.append(token_rep)
        except KeyError:
            print("KeyError: " + w)

    return all_sentence_idx, all_word_idx, all_tokens, all_token_rep, all_labels


def generate_csv_file(all_sentence_idx, all_word_idx, all_tokens, all_token_rep, all_labels, output_file):
    df = pd.DataFrame(
        {'token': all_tokens,
         'line_idx': all_sentence_idx,
         'position_idx': all_word_idx,
         'embedding': all_token_rep,
         'labels': all_labels
        })
    df.to_csv(output_file, index=False, sep='\t')


def main():
    parser = ArgumentParser()
    parser.add_argument("--explanationFile", type=str, default="",
                        help="Path to the explanation file")
    parser.add_argument("--datasetFile", type=str, default="",
                        help="Path to the dataset file")
    parser.add_argument("--outputFile", type=str, default="",
                        help="Path to the output file")
    args = parser.parse_args()

    explanationFile = fileRead(args.explanationFile, 1)
    dataset_map = read_dataset_file(args.datasetFile)
    all_sentence_idx, all_word_idx, all_tokens, all_token_rep, all_labels = match_to_representation(explanationFile, dataset_map)
    generate_csv_file(all_sentence_idx, all_word_idx, all_tokens, all_token_rep, all_labels, args.outputFile)


if __name__ == "__main__":
    main()










