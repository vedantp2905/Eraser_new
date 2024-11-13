import argparse

import pandas as pd


def read_data(input_file1, input_file2):
    # read the text file
    with open(input_file1, 'r') as f:
        sentence1 = f.readlines()

    with open(input_file2, 'r') as f:
        sentence2 = f.readlines()

    return sentence1, sentence2


def convert_to_tsv(sentence1, sentence2, save_filename):
    # dataframe to save the data
    df = pd.DataFrame(columns=["sentence1", "sentence2"])

    df["sentence1"] = sentence1
    df["sentence2"] = sentence2

    # strip the \n at the end of each sentence
    df["sentence1"] = df["sentence1"].str.strip()
    df["sentence2"] = df["sentence2"].str.strip()

    # save the dataframe to tsv file
    df.to_csv(save_filename, sep="\t", index=False, header=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file1', type=str, default="data/stsb", help="sentence 1 file name")
    parser.add_argument('--input-file2', type=str, default="data/stsb", help="sentence 2 file name")
    parser.add_argument('--save-file', type=str, default="data/", help="save file name")

    args = parser.parse_args()

    sentence1, sentence2 = read_data(args.input_file1, args.input_file2)
    convert_to_tsv(sentence1, sentence2, args.save_file)


if __name__ == "__main__":
    main()