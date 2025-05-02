import argparse
import pandas as pd
import ast


def get_salient_words(filename, method='top-1', attribution_mass=None):
    # top-1 means the most salient word, top-k means the top k salient words with a certain attribution mass
    assert method in ['top-1', 'top-k']

    # read the csv in the dataframe
    df = pd.read_csv(filename)

    # get the saliencies
    saliencies = df['saliencies'].tolist()

    # go through the saliencies which are in the format (token, score) and get the salient words
    salient_words = []
    sentence_ids = []
    predicted_classes = []
    for sentence_idx, saliency in enumerate(saliencies):
        saliency = ast.literal_eval(saliency)

        # sort the saliency in descending order and save original index
        sorted_saliency = sorted(enumerate(saliency[:-1]), key=lambda x: x[1][1], reverse=True)

        # get the salient words
        if method == 'top-1':
            # get the most salient word and save in the format (word, index)
            salient_words.append((sorted_saliency[0][1][0], sorted_saliency[0][0]-1))

            sentence_ids.append(sentence_idx)
            predicted_classes.append(df['predicted_class'].tolist()[sentence_idx])
        elif method == 'top-k':
            if attribution_mass is None:
                raise ValueError("attribution_mass must be specified when using top-k method")
            else:
                # get the top k salient words with a certain attribution mass
                attribution_sum = 0
                for (w, s) in sorted_saliency:
                    attribution_sum += s[1]
                    salient_words.append((s, w-1))
                    sentence_ids.append(sentence_idx)
                    predicted_classes.append(df['predicted_class'].tolist()[sentence_idx])

                    if attribution_sum >= attribution_mass:
                        break

    return salient_words, sentence_ids, predicted_classes


def generate_explanation_file(important_tokens, sentence_id, prediction_class):
    explanation_content = []

    for i in range(len(important_tokens)):
        explanation_content.append(str(prediction_class[i]) + " " + str(important_tokens[i][1]) + " " + str(sentence_id[i]))

    return explanation_content


def save_explanation(save_file, explanation_content):
    with open(save_file, "w") as fp:
        for line in explanation_content:
            fp.write(line + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file")
    parser.add_argument("save_file")
    parser.add_argument("method", default='top-1', choices=['top-1', 'top-k'])
    parser.add_argument("--attribution_mass", type=float, default=None)

    args = parser.parse_args()

    important_tokens, sentence_id, prediction_class = get_salient_words(args.input_file, args.method, args.attribution_mass)
    print("Finishing getting important tokens")

    explanation_content = generate_explanation_file(important_tokens, sentence_id, prediction_class)
    print("Finishing generating explanation file")

    save_explanation(args.save_file, explanation_content)
    print("Finishing saving explanation file")


if __name__ == "__main__":
    main()





