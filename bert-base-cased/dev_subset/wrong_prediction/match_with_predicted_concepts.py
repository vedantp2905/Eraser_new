import pandas as pd
import ast
from argparse import ArgumentParser


def load_wrong_prediction_data(wrong_prediction_files):
    wrong_prediction = pd.read_csv(wrong_prediction_files)
    return wrong_prediction

def load_predicted_concepts(predicted_concepts_file):
    predicted_concepts = pd.read_csv(predicted_concepts_file, sep='\t')

    predicted_concepts_map = {}
    for i in range(len(predicted_concepts)):
        line_idx = predicted_concepts['line_idx'][i]
        word_idx = predicted_concepts['position_idx'][i]
        token = str(word_idx) + '_' + str(line_idx)
        predicted_concepts_map[token] = (predicted_concepts['Token'][i], predicted_concepts['Top 2'][i])

    return predicted_concepts_map


def match_with_predicted_concepts(wrong_prediction, predicted_concepts_map):
    explanations_list = wrong_prediction['explanation'].tolist()

    predicted_concepts_list = []
    sentence_idx_list = []
    word_idx_list = []
    token_list = []
    predicted_labels_list = []
    gold_labels_list = []

    explanations_list = [ast.literal_eval(explanation) for explanation in explanations_list]
    for i, explanations in enumerate(explanations_list):
        for current_explanation in explanations:
            label = current_explanation.split(' ')[0]
            sentence_idx = current_explanation.split(' ')[2]
            word_idx = current_explanation.split(' ')[1]
            predicted_concepts = predicted_concepts_map[word_idx+'_'+sentence_idx]

            sentence_idx_list.append(sentence_idx)
            word_idx_list.append(word_idx)
            predicted_concepts_list.append(predicted_concepts[1])
            predicted_labels_list.append(label)
            token_list.append(predicted_concepts[0])
            gold_labels_list.append(wrong_prediction['gold_label'][i])

    return sentence_idx_list, word_idx_list, predicted_concepts_list, gold_labels_list, predicted_labels_list, token_list


def write_to_csv(sentence_idx_list, word_idx_list, predicted_concepts_list, gold_labels_list, predicted_labels_list, token_list, output_file):
    # create a new dataframe
    df = pd.DataFrame(
        {'token': token_list,
         'line_idx': sentence_idx_list,
         'position_idx': word_idx_list,
         'predicted_concepts': predicted_concepts_list,
         'gold_labels': gold_labels_list,
         'predicted_labels': predicted_labels_list
        })

    df.to_csv(output_file, index=False, sep='\t')


def main():
    parser = ArgumentParser()
    parser.add_argument("--wrong_prediction_files", type=str)
    parser.add_argument("--predicted_concepts_file", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    wrong_prediction = load_wrong_prediction_data(args.wrong_prediction_files)
    predicted_concepts_map = load_predicted_concepts(args.predicted_concepts_file)
    sentence_idx_list, word_idx_list, predicted_concepts_list, gold_labels_list, predicted_labels_list, token_list = match_with_predicted_concepts(wrong_prediction, predicted_concepts_map)
    write_to_csv(sentence_idx_list, word_idx_list, predicted_concepts_list, gold_labels_list, predicted_labels_list, token_list, args.output_file)


if __name__ == '__main__':
    main()


