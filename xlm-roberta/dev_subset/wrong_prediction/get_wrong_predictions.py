import pandas as pd
import json
from argparse import ArgumentParser


def load_preidction_file(fname):
    df = pd.read_csv(fname, sep='\t')
    return df


def load_gold_label_file(fname):
    with open(fname, "r") as f:
        data = json.load(f)

    labels = []
    sentences = []
    for line in data:
        labels.append(line['label'])
        sentences.append(line['sentence'])
    return labels, sentences


def load_explanation_file(fname):
    lines = []
    print("Reading file: " + fname)
    with open(fname) as f:
        for line in f:
            line = line.rstrip('\r\n')
            lines.append(line)

    f.close()
    return lines


def get_explanation_by_sentence_idx(explanation_file, sentence_idx):
    explanation = []
    for w in explanation_file:
        if w.split(" ")[2] == sentence_idx:
            explanation.append(w)

    return explanation


def get_wrong_predictions(gold_labels, gold_sentence, predicted_df, explanation_file):
    predicted_labels = predicted_df['labels'].tolist()

    wrong_predictions_idx = []
    wrong_predictions_sentence = []
    explanations = []
    gold_labels_list = []
    predicted_labels_list = []
    for i, gold_label in enumerate(gold_labels):
        if gold_label != predicted_labels[i]:
            wrong_predictions_idx.append(i)
            wrong_predictions_sentence.append(gold_sentence[i])
            explanations.append(get_explanation_by_sentence_idx(explanation_file, str(i)))
            gold_labels_list.append(gold_label)
            predicted_labels_list.append(predicted_labels[i])

    print("% of wrong predictions: " + str(len(wrong_predictions_idx) / len(gold_labels) * 100))

    return wrong_predictions_idx, wrong_predictions_sentence, explanations, gold_labels_list, predicted_labels_list


# write the wrong predictions information into a csv file
def write_wrong_predictions(wrong_predictions_idx, wrong_predictions_sentence, explanations, gold_labels_list,
                            predicted_labels_list, fname):
    df = pd.DataFrame(
        {'idx': wrong_predictions_idx, 'sentence': wrong_predictions_sentence, 'explanation': explanations,
         'gold_label': gold_labels_list, 'predicted_label': predicted_labels_list})
    df.to_csv(fname, index=False)


def main():

    parser = ArgumentParser()
    parser.add_argument("--gold_label_file", type=str)
    parser.add_argument("--predicted_file", type=str)
    parser.add_argument("--explanation_file", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    gold_label_file = args.gold_label_file
    predicted_file = args.predicted_file
    explanation_file = args.explanation_file

    # load the gold label file
    gold_label, gold_sentence = load_gold_label_file(gold_label_file)

    # load the predicted file
    predicted_df = load_preidction_file(predicted_file)

    # load the explanation file
    explanation_file = load_explanation_file(explanation_file)


    # get the wrong predictions
    wrong_predictions_idx, wrong_predictions_sentence, explanations, gold_labels_list, predicted_labels_list = get_wrong_predictions(
        gold_label, gold_sentence, predicted_df, explanation_file)

    # write the wrong predictions information into a csv file
    write_wrong_predictions(wrong_predictions_idx, wrong_predictions_sentence, explanations, gold_labels_list,
                            predicted_labels_list, args.output_file)




if __name__ == "__main__":
    main()




