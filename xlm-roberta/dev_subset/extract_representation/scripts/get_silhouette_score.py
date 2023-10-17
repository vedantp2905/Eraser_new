import argparse
import pickle

import numpy as np
from sklearn.metrics import silhouette_score


def load_data(point_file):
    points = np.load(point_file)
    return points


def load_model(model_path):
    classifier = pickle.load(open(model_path, 'rb'))
    return classifier


def get_silhouette_score(points, classifier):
    score = silhouette_score(points, classifier.labels_, metric='euclidean')
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--point-file",
                        "-p",
                        help="output point file with complete path")
    parser.add_argument('--model-file-path',
                       type=str,
                       default='LogisticRegression',
                       help='the classifier to use')
    args = parser.parse_args()

    points = load_data(args.point_file)
    classifier = load_model(args.model_file_path)
    score = get_silhouette_score(points, classifier)
    print('silhouette_score:', score)


if __name__ == '__main__':
    main()
