import argparse
import ast
import os
import random

import pandas as pd


def read_csv(file_path, layer, only_CLS):
    """
    Read the clusters with embedding values (csv file) and return the dataframe

    Parameters
    ----------
    file_path : str
        The path of the csv file

    layer : int
        The layer of the csv file

    Returns
    -------
    df : pandas.DataFrame
        The dataframe of the csv file

    """
    df = pd.read_csv(file_path + 'clusters-map' + str(layer) + '.csv', sep='\t')

    if only_CLS:
        df = df[df['position_idx'] == -1]
        df = df.reset_index(drop=True)

    return df


# split the dataframe into train and validation without balancing the label
def split_dataset(df, validation_size, random_seed, train_dataset_save_path, validation_dataset_save_path, layer):
    """
    Split the dataframe into train and validation without balancing the sentiment label

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe of the csv file

    validation_size : float
        The size of the validation dataset

    random_seed : int
        The random seed

    validation_dataset_save_path : str
        The path to save the validation dataset

    layer : int
        The layer of the csv file

    Returns
    -------
    train_df : pandas.DataFrame
        The dataframe of the train dataset

    validation_df : pandas.DataFrame
        The dataframe of the validation dataset

    ids : list
        The list of the position and line value of each token in the validation dataset
    """

    ids = []

    validation_size = len(df) * validation_size
    validation_size = int(validation_size)

    random.seed(random_seed)

    random_number = random.sample(range(0, len(df)), validation_size)
    validation_df = df.iloc[random_number]
    train_df = df.drop(random_number)

    # go through validation_df and save the position and line value of each one
    for i in range(len(validation_df)):
        ids.append([validation_df.iloc[i]['line_idx'], validation_df.iloc[i]['position_idx']])

    validation_df.to_csv(validation_dataset_save_path + 'validation_df_' + str(layer) + '.csv', sep='\t', index=False)

    train_df.to_csv(train_dataset_save_path + 'train_df_' + str(layer) + '.csv', sep='\t', index=False)

    return ids


def read_ids(id_filename):
    """
    Read the ids file and return the list of the position and line value of each token in the validation dataset

    Parameters
    ----------
    id_filename : str
        The name of the file that contains the position and line value of each token in the validation dataset

    Returns
    -------
    ids : list
        The list of the position and line value of each token in the validation dataset
    """
    with open(id_filename, 'r') as f:
        ids = [ast.literal_eval(line.rstrip()) for line in f]
    return ids


def check_test_data(df, ids, train_dataset_save_path, validation_dataset_save_path, layer):
    """
    Go through dataframe and check if the line and position of each token is in the i   ds list, if yes, it is validation data

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe of the csv file

    ids : list
        The list of the position and line value of each token in the validation dataset

    validation_dataset_save_path : str
        The path to save the validation dataset

    layer : int
        The layer of the csv file

    Returns
    -------
    validation_df : pandas.DataFrame
        The dataframe of the validation dataset

    train_df : pandas.DataFrame
        The dataframe of the train dataset
    """

    index = []

    for i in range(len(df)):
        if [df.iloc[i]['line_idx'], df.iloc[i]['position_idx']] in ids:
            index.append(i)

    validation_df = df.iloc[index]
    train_df = df.drop(index)

    validation_df.to_csv(validation_dataset_save_path + 'validation_df_' + str(layer) + '.csv', sep='\t', index=False)

    train_df.to_csv(train_dataset_save_path + 'train_df_' + str(layer) + '.csv', sep='\t', index=False)


def main():
    parse = argparse.ArgumentParser()

    parse.add_argument('--random_seed',
                       type=int,
                       default=42,
                       help='random seed number')

    parse.add_argument('--file_path',
                       type=str,
                       default='./clusters_csv/',
                       help='cluster value file path')

    parse.add_argument('--layer',
                    #    type=int,
                       default=12,
                       help='the selected layer number')

    parse.add_argument('--validation_size',
                       type=float,
                       default=0.1,
                       help='set the validation size (a percentage like 0.1)')

    parse.add_argument('--is_first_file',
                       action="store_true",
                       help='is first file to predict or not')

    parse.add_argument('--validation_dataset_save_path',
                       type=str,
                       default='./lr_validation_dataset/',
                       help='save validation dataset path')

    parse.add_argument('--train_dataset_save_path',
                       type=str,
                       default='./lr_train_dataset/',
                       help='save validation dataset path')

    parse.add_argument('--id_save_filename',
                       type=str,
                       default='lr_id.txt',
                       help='save id data path')

    parse.add_argument('--only_CLS',
                       action="store_true",
                       help='only use CLS token or not')

    args = parse.parse_args()

    df = read_csv(args.file_path, args.layer, args.only_CLS)
    print("Finished reading the csv file")

    if not os.path.exists(args.validation_dataset_save_path):
        os.makedirs(args.validation_dataset_save_path)

    if not os.path.exists(args.train_dataset_save_path):
        os.makedirs(args.train_dataset_save_path)

    if args.is_first_file:
        ids = split_dataset(df, args.validation_size, args.random_seed, args.train_dataset_save_path,
                                                                args.validation_dataset_save_path, args.layer)

        with open(args.id_save_filename, 'w') as f:
            for item in ids:
                f.write("%s\n" % item)
    else:
        ids = read_ids(args.id_save_filename)
        check_test_data(df, ids, args.train_dataset_save_path, args.validation_dataset_save_path, args.layer)

    print("Finished splitting the dataset")


if __name__ == '__main__':
    main()