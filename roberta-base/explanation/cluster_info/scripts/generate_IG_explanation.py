import argparse
import pandas as pd
import ast


def get_important_tokens(filename):
    # read the csv in the dataframe
    df = pd.read_csv(filename)

    # get the saliencies
    saliencies = df['saliencies'].tolist()

    # go through the saliencies which are in the format (token, score) and get the top 1 token
    important_tokens = []
    for saliency in saliencies:

        saliency = ast.literal_eval(saliency)

        max_value = 0
        max_index = 0

        for i, (w, s) in enumerate(saliency[:-1]):
            if s > max_value:
                max_value = s
                max_index = i

        important_tokens.append((saliency[max_index][0], max_index-1))

        # find the first negative saliency and the position of the token
        # neg_max_index = 0
        # neg_max = 0
        # for i, (w, s) in enumerate(saliency[:-1]):
        #     if s < 0:
        #         neg_max_index = i
        #         neg_max = s
        #         break

        # for i, (w, s) in enumerate(saliency[:-1]):
        #     if s < 0 and s > neg_max:
        #         neg_max_index = i
        #         neg_max = s

        # important_tokens.append((saliency[neg_max_index][0], neg_max_index-1))

    return important_tokens, df['sentence_id'].tolist(), df['predicted_class'].tolist()


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

    args = parser.parse_args()

    important_tokens, sentence_id, prediction_class = get_important_tokens(args.input_file)
    print("Finishing getting important tokens")

    explanation_content = generate_explanation_file(important_tokens, sentence_id, prediction_class)
    print("Finishing generating explanation file")

    save_explanation(args.save_file, explanation_content)
    print("Finishing saving explanation file")


if __name__ == "__main__":
    main()
