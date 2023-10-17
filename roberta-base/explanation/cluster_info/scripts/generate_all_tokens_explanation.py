import json
from argparse import ArgumentParser


def fileRead(fname, flag):
    lines = []

    if (flag == 0):
        with open(fname, "r") as f:
            data = json.load(f)

        for line in data:
            line = line.rstrip('\r\n')
            lines.append(line)
    else:
        print("Reading file: " + fname)
        with open(fname) as f:
            for line in f:
                line = line.rstrip('\r\n')
                lines.append(line)

    f.close()
    return lines


def store_labels(lines):
    labels = []
    for line in lines:
        explanation_label = line.split(" ")[0]
        labels.append(explanation_label)
    return labels


def clusterRead(fname, labels):
    word_explanations = []

    with open(fname) as f:
        for line in f:
            line = line.rstrip('\r\n')
            parts = line.split("|||")

            sent_idx = int(parts[2])
            word_idx = int(parts[3])
            label = labels[sent_idx]

            word_explanations.append(label + " " + str(word_idx) + " " + str(sent_idx))
    return word_explanations


def add_word_explanation(fname, word_explanations):
    # append the word explanation to the end of the file
    with open(fname, "w") as f:
        for line in word_explanations:
            f.write(line + "\n")
    f.close()


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--clusterFile", dest="c",
                        help="Cluster File", metavar="FILE")
    parser.add_argument("-e", "--explanationFile", dest="e",
                        help="Explanation File", metavar="FILE")
    parser.add_argument("-s", "--saveFile", dest="s",
                        help="New Explanation File save path", metavar="FILE")

    args = parser.parse_args()

    explanationFile = fileRead(args.e, 1)
    labels = store_labels(explanationFile)
    word_explanations = clusterRead(args.c, labels)
    add_word_explanation(args.s, word_explanations)


if __name__ == "__main__":
    main()


