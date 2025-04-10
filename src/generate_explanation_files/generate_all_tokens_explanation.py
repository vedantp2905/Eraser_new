import json
from argparse import ArgumentParser


def fileRead(fname, flag):
    lines = []
    print("Reading file: " + fname)
    with open(fname) as f:
        for line in f:
            line = line.rstrip('\r\n')
            lines.append(line)
    return lines


def store_labels(lines):
    labels = []
    for line in lines:
        explanation_label = line.split(" ")[0]
        labels.append(explanation_label)
    return labels


def read_dataset_file(fname):
    sentence_idx = []
    word_idx = []
    
    print(f"Reading cluster file: {fname}")
    with open(fname, 'r') as f:
        for line in f:
            parts = line.strip().split("|||")
            if len(parts) >= 4:  # We need at least 4 parts for token, cluster_id, sentence_id, word_id
                sentence_idx.append(parts[2])  # sentence_id is at index 2
                word_idx.append(parts[3])      # word_id is at index 3
    
    print(f"Found {len(sentence_idx)} tokens")
    return sentence_idx, word_idx


def generate_word_explanation(sentence_idx, word_idx, labels):
    word_explanations = []
    for i in range(len(sentence_idx)):
        try:
            # Convert indices to int for validation
            sent_idx = int(sentence_idx[i])
            if sent_idx < len(labels):
                word_explanation = f"{labels[sent_idx]} {word_idx[i]} {sentence_idx[i]}"
                word_explanations.append(word_explanation)
            else:
                print(f"Warning: sentence index {sent_idx} out of range. Max index is {len(labels)-1}")
        except ValueError as e:
            print(f"Warning: invalid sentence index {sentence_idx[i]}: {str(e)}")
            continue
    return word_explanations


def write_word_explanation(fname, word_explanations):
    print(f"Writing {len(word_explanations)} explanations to {fname}")
    with open(fname, 'w') as outfile:
        for word_explanation in word_explanations:
            outfile.write(word_explanation + '\n')


def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--datasetFile", dest="c",
                       help="Cluster File", metavar="FILE")
    parser.add_argument("-e", "--explanationFile", dest="e",
                       help="Explanation File", metavar="FILE")
    parser.add_argument("-s", "--saveFile", dest="s",
                       help="New Explanation File save path", metavar="FILE")

    args = parser.parse_args()

    print("\nStep 1: Reading explanation file")
    explanationFile = fileRead(args.e, 1)
    labels = store_labels(explanationFile)
    print(f"Found {len(labels)} labels")

    print("\nStep 2: Reading cluster file")
    sentence_idx, word_idx = read_dataset_file(args.c)
    print(f"Found {len(sentence_idx)} sentence indices and {len(word_idx)} word indices")

    print("\nStep 3: Generating word explanations")
    word_explanations = generate_word_explanation(sentence_idx, word_idx, labels)
    print(f"Generated {len(word_explanations)} explanations")

    print("\nStep 4: Writing results")
    write_word_explanation(args.s, word_explanations)
    print("Done!")


if __name__ == "__main__":
    main()


