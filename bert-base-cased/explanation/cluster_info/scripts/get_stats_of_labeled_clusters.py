import argparse
import json


def clusterLabelRead(fname):
    with open(fname) as f:
        data = json.load(f)
    return data


def get_number_of_labledCluster(clusterLabel):
    labelCount = {}
    for key, value in clusterLabel.items():
        if value in labelCount:
            labelCount[value] += 1
        else:
            labelCount[value] = 1
    return labelCount

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("-c", "--clusterLabelFile", dest="c", help="Cluster Label File", metavar="FILE")
    args = parse.parse_args()

    clusterLabel = clusterLabelRead(args.c)
    labelCount = get_number_of_labledCluster(clusterLabel)
    print(labelCount)


if __name__ == "__main__":
    main()