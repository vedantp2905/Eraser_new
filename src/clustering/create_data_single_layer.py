import argparse
import json
import sys

from collections import Counter
from tqdm import tqdm

sys.path.append('NeuroX')
import neurox.data.loader as data_loader


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--text-file', type=str, required=True)
    parser.add_argument('--activation-file', type=str, required=True)
    parser.add_argument('--output-prefix', type=str, required=True)
    parser.add_argument('--is-save-sep', action='store_true', required=False)
    parser.add_argument('--sentence-tag', type=str, default="[CLS]",
                        help="sentence tag used in the tokenizer ([CLS] or <s>)")

    args = parser.parse_args()

    print("Loading activations...")
    activations, num_layers = data_loader.load_activations(f'{args.activation_file}', num_neurons_per_layer=768)
    print("Loading tokens...")
    tokens = data_loader.load_data(
        f'{args.text_file}',
        f'{args.text_file}',
        activations,
        1000
    )

    print("Preparing dataset...")
    sentences = []
    labels = []
    selected_tokens = Counter()
    token_dataset = []

    for line_idx, label_line in tqdm(enumerate(tokens['target'])):
        sentences.append(" ".join(tokens['source'][line_idx]))
        labels.append(" ".join(label_line))
        for label_idx, label in enumerate(label_line):
            token = tokens['source'][line_idx][label_idx]

            start_tag = "[CLS]"
            end_tag = "[SEP]"
            if args.sentence_tag == "<s>":
                start_tag = "<s>"
                end_tag = "</s>"

            if (token == end_tag and not args.is_save_sep) or (
                    token != start_tag and token != end_tag and token[0] == '[' and token[-1] == ']'):
                continue
            else:
                selected_tokens[token] += 1
                token_acts = activations[line_idx][label_idx, :]
                token_acts = token_acts.tolist()

                if token == start_tag:
                    token = start_tag + str(line_idx)
                elif token == end_tag:
                    if label_idx == len(label_line) - 1:
                        token = end_tag + str(line_idx) + "_last"
                    else:
                        if args.sentence_tag == "<s>":
                            if tokens['source'][line_idx][label_idx-1] == "</s>":
                                token = end_tag + str(line_idx) + "_middle2"
                            else:
                                token = end_tag + str(line_idx) + "_middle1"
                        else:
                            token = end_tag + str(line_idx) + "_middle"

                final_tok_rep = f'{token}|||{selected_tokens[token]}|||{len(sentences) - 1}|||{label_idx - 1}'
                token_dataset.append((final_tok_rep, token_acts))

    print("Writing datasets...")
    with open(f'{args.output_prefix}-sentences.json', 'w', encoding='utf-8') as fp:
        json.dump(sentences, fp, ensure_ascii=False)

    with open(f'{args.output_prefix}-labels.json', 'w') as fp:
        json.dump(labels, fp)

    with open(f'{args.output_prefix}-dataset.json', 'w') as fp:
        json.dump(token_dataset, fp)


if __name__ == '__main__':
    main()
