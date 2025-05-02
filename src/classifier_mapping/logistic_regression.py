import argparse
import ast
import os

import dill as pickle
import numpy as np

import pandas as pd
from sklearn.linear_model import LogisticRegression


def train_classifier(train_df, classifier):
    """
    Train the Logistic Regression Classifier

    Parameters
    ----------
    train_df : pandas.DataFrame
        The dataframe of the train dataset

    classifier : sklearn.linear_model.LogisticRegression
        The classifier to train

    Returns
    -------
    classifier : sklearn.linear_model.LogisticRegression
        The trained classifier
    """
    # train
    X_train = train_df['embedding']
    X_train = [ast.literal_eval(val) for val in X_train]
    y_train = train_df['cluster_idx']

    # train the classifier
    classifier.fit(X_train, y_train)

    return classifier


def save_classifier(classifier, layer, output_Path):
    """
    Save the Logistic Regression Classifier

    Parameters
    ----------
    classifier : sklearn.linear_model.LogisticRegression
        The classifier to save

    layer : int
        The layer of the csv file

    output_Path : str
        The path to save the classifier

    Returns
    -------
    None
    """
    # save the classifier
    filename = output_Path + 'layer_' + str(layer) + '_classifier.pkl'

    with open(filename, 'wb') as file:
        pickle.dump(classifier, file)


def format_predictions(classifier, X, df):
    """
    Format the predictions of the Logistic Regression Classifier

    Parameters
    ----------
    classifier : sklearn.linear_model.LogisticRegression
        The classifier to save

    X : list
        The list of the values

    df : pandas.DataFrame
        The dataframe of the dataset
    Returns
    -------
    pred_df : pandas.DataFrame
        The predictions of the classifier
    """
    # get classes
    classes = classifier.classes_

    # get the probability of each class
    probs = classifier.predict_proba(X)

    # make a dataframe for the predictions
    pred_df = pd.DataFrame(columns=['Token', 'Top 1', 'Top 2', 'Top 5'])

    # tokens
    tokens = df['token'].values

    # get the top 1, 2, 5 predictions
    top1 = []
    top2 = []
    top5 = []
    for i in range(len(probs)):
        # sort the probabilities in increasing order
        sorted_index = np.argsort(probs[i])

        # get the top 1 prediction
        top1.append(classes[sorted_index[-1]])

        # get all top 2 predictions
        top2.append(classes[sorted_index[-2:]])

        # get all top 5 predictions
        top5.append(classes[sorted_index[-5:]])

    # add the predictions to the dataframe
    pred_df['Token'] = tokens
    pred_df['Top 1'] = top1
    pred_df['Top 2'] = top2
    pred_df['Top 5'] = top5

    return pred_df


def validate_classifier(validate_df, classifier, layer, output_Path):
    """
    Validate the Logistic Regression Classifier

    Parameters
    ----------
    validate_df : pandas.DataFrame
        The dataframe of the validate dataset

    classifier : sklearn.linear_model.LogisticRegression
        The classifier to validate

    layer : int
        The layer of the csv file

    output_Path : str
        The path to save the predictions

    Returns
    -------
    accuracy : float
        The accuracy of the classifier
    """
    # validate
    X_validate = validate_df['embedding']
    X_validate = [ast.literal_eval(val) for val in X_validate]
    y_validate = validate_df['cluster_idx']

    # validate the classifier
    accuracy = classifier.score(X_validate, y_validate)

    df = format_predictions(classifier, X_validate, validate_df)

    # add the Actual column at the index 1
    df.insert(1, 'Actual', y_validate.values)

    # save the predictions
    df.to_csv(output_Path + 'predictions_layer_' + str(layer) + '.csv', sep='\t', index=False)

    return accuracy


def predict_classifier(test_df, classifier, layer, output_Path):
    """
    Predict the Logistic Regression Classifier

    Parameters
    ----------
    test_df : pandas.DataFrame
        The dataframe of the test dataset

    classifier : sklearn.linear_model.LogisticRegression
        The classifier to predict

    layer : int
        The layer of the csv file

    Returns
    -------
    predictions : pandas.DataFrame
        The predictions of the classifier
    """
    # predict
    X_test = test_df['embedding']
    X_test = [ast.literal_eval(val) for val in X_test]

    # predict the classifier
    classifier.predict(X_test)

    df = format_predictions(classifier, X_test, test_df)

    df.to_csv(output_Path + 'predictions_layer_' + str(layer) + '.csv', sep='\t', index=False)


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--train_file_path',
                       type=str,
                       help='train cluster csv file path')

    parse.add_argument('--validate_file_path',
                       type=str,
                       help='validate cluster csv file path')

    parse.add_argument('--test_file_path',
                       type=str,
                       help='test cluster csv file path')

    parse.add_argument('--classifier_file_path',
                        type=str,
                        default='LogisticRegression',
                        help='the classifier to use')

    parse.add_argument('--layer',
                    #    type=int,
                       default=12,
                       help='the selected layer number')

    parse.add_argument('--save_path',
                       type=str,
                       help='save classification result path')

    parse.add_argument('--do_train',
                       action="store_true",
                       help='whether to train the classifier')

    parse.add_argument('--do_validate',
                        action="store_true",
                        help='whether to validate the classifier')

    parse.add_argument('--do_predict',
                        action="store_true",
                        help='whether to predict the classifier')

    parse.add_argument('--load_classifier_from_local',
                        action="store_true",
                        help='whether to load the classifier from local')

    args = parse.parse_args()

    if args.load_classifier_from_local:
        classifier = pickle.load(open(args.classifier_file_path, 'rb'))
    else:
        classifier = LogisticRegression()

    if args.do_train:
        train_df = pd.read_csv(args.train_file_path, sep='\t')
        classifier = train_classifier(train_df, classifier)

        save_path = args.save_path + '/model/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_classifier(classifier, args.layer, save_path)

    if args.do_validate:
        validate_df = pd.read_csv(args.validate_file_path, sep='\t')

        save_path = args.save_path + '/validate_predictions/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        accuracy = validate_classifier(validate_df, classifier, args.layer, save_path)
        print('Accuracy: ', accuracy)

    if args.do_predict:
        test_df = pd.read_csv(args.test_file_path, sep='\t')

        save_path = args.save_path + '/test_predictions/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        predict_classifier(test_df, classifier, args.layer, save_path)


if __name__ == '__main__':
    main()



