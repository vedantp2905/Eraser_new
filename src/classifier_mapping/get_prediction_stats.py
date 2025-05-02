""" Get the prediction stats for the classification result

This script is used to get the prediction stats for the classification result.

"""

import pandas as pd
import argparse


# read the file 'predictions_layer_0.csv'
def read_csv(file_path, layer):
    """
    Read the Classification result (csv file)

    Parameters:
    ----------
    file_path: str
        The path of the csv file

    layer: str
        The layer of the csv file

    Returns:
    -------
    df: pandas.DataFrame
        The dataframe of the csv file
    """
    df = pd.read_csv(file_path + 'predictions_layer_' + layer + '.csv', sep='\t')
    return df


def check_predictions(df):
    """
    Check the predictions with top 5 prediction results and the accuracy of each top 1, top 2, top 5

    Parameters:
    ----------
    df: pandas.DataFrame
        The dataframe of the csv file

    """

    top1 = 0
    top2 = 0
    top5 = 0
    no_match = 0

    for i in range(len(df)):
        if str(df['Actual'][i]) == str(df['Top 1'][i]):
            top1 += 1

        if str(df['Actual'][i]) in df['Top 2'][i]:
            top2 += 1

        if str(df['Actual'][i]) in df['Top 5'][i]:
            top5 += 1
        else:
            no_match += 1

    print("Layer: ", args.layer)
    print("Total Tokens: ", len(df))
    print("Match with Top 1: ", top1, ', Percentage: ', top1 / len(df) * 100, '%')
    print("Match with Top 2: ", top2, ', Percentage: ', top2 / len(df) * 100, '%')
    print("Match with Top 5: ", top5, ', Percentage: ', top5 / len(df) * 100, '%')
    print("No Match: ", no_match, ', Percentage: ', no_match / len(df) * 100, '%' )
    print("---------------------------------------------------")


def get_all_layer_stats():
    """
    Get the prediction stats for all layers
    """
    for i in range(13):
        args.layer = str(i)
        df = read_csv(args.file_path, args.layer)
        check_predictions(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--layer', type=str, default=12, help='layer number (can have multiple layers separated by '
                                                              'comma)')
    parser.add_argument('--all_layer_stats', action='store_true', help='get all layer stats')
    parser.add_argument('--file_path', type=str, default='./LR_classification/', help='file path')

    args = parser.parse_args()

    if args.all_layer_stats:
        get_all_layer_stats()
    else:
        choose_layer = args.layer.split(',')
        for layer in choose_layer:
            args.layer = layer
            df = read_csv(args.file_path, args.layer)
            check_predictions(df)
