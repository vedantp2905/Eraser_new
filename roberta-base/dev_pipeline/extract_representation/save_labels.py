# read the json file and save the labels into txt file
import argparse
import json


def read_json(textFile, filename):
    with open(textFile, 'r') as f:
        data = json.load(f)

    with open(filename, 'w') as f:
        for i in range(len(data)):
            gold_label = data[i]['label']
            f.write(str(gold_label) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--text-file', type=str, required=True)
    parser.add_argument('--dataset-name', type=str, help="name of the dataset")

    args = parser.parse_args()

    read_json(args.text_file, args.dataset_name)






