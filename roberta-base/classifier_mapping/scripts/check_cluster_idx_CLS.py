# read csv file by pandas
import argparse

import pandas as pd


def load_CSV(filename):
    data = pd.read_csv(filename, sep='\t')
    return data


def check_cluster_id(data):
    # print the cluster id for the token has the position_idx -1
    CLS_cluster_id = []
    for i in range(len(data)):
        if data['position_idx'][i] == -1:
            CLS_cluster_id.append(data['cluster_idx'][i])

    # remove the duplicate cluster id
    CLS_cluster_id = list(set(CLS_cluster_id))
    
    return CLS_cluster_id


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--filename', type=str, default='clusters_csv_train/clusters-map12.csv')
    args = parse.parse_args()
    
    data = load_CSV(args.filename)
    CLS_cluster_id = check_cluster_id(data)
    print('# of clusters contain [CLS] tokens:', len(CLS_cluster_id))
    print('cluster_idx:', CLS_cluster_id)


if __name__ == '__main__':
    main()
