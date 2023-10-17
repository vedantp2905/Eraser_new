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
        with open(fname) as f:
            for line in f:
                line = line.rstrip('\r\n')
                lines.append(line)

    f.close()
    return lines


def clusterLabelRead(fname):
    with open(fname) as f:
        data = json.load(f)
    return data


def clusterRead(fname, explanationMap):
    words = []
    words_idx = []
    cluster_idx = []
    sent_idx = []

    with open(fname) as f:
        for line in f:
            line = line.rstrip('\r\n')
            parts = line.split("|||")

            cluster_id = int(parts[4])
            word_id = int(parts[3])
            sent_id = int(parts[2])

            token_pos = str(word_id) + "_" + str(sent_id)
            if token_pos in explanationMap:
                cluster_idx.append(cluster_id)
                words_idx.append(word_id)
                sent_idx.append(sent_id)
                words.append(parts[0])

    return words, words_idx, sent_idx, cluster_idx


parser = ArgumentParser()
parser.add_argument("-c", "--clusterFile", dest="c",
                    help="Cluster File", metavar="FILE")
parser.add_argument("-e", "--explanationFile", dest="e",
                    help="Explanation File", metavar="FILE")
parser.add_argument("-l", "--clusterLabelFile", dest="l",
                    help="Explanation File", metavar="FILE")
parser.add_argument("-p", "--percentage", dest="p",
                    help="Percentage", metavar="FILE")

args = parser.parse_args()

explanationMap = {}

explanationFile = fileRead(args.e, 1)

for i, w in enumerate(explanationFile):
    words = w.split(" ")

    t = words[1] + "_" + words[2]
    explanationMap[t] = words[0]

clusterLabel = clusterLabelRead(args.l)

words, words_idx, sent_idx, cluster_idx = clusterRead(args.c, explanationMap)

clusterMap = {}
prev = cluster_idx[0]
for i, w in enumerate(cluster_idx):

    if (prev == cluster_idx[i]):
        thisEntry = str(words_idx[i]) + "_" + str(sent_idx[i])
        thisLabel = explanationMap[thisEntry]
        prev = cluster_idx[i]
    else:
        prev = cluster_idx[i]
        thisEntry = str(words_idx[i]) + "_" + str(sent_idx[i])
        thisLabel = explanationMap[thisEntry]

    clusterMap[thisEntry] = cluster_idx[i]

total = 0
hit = 0
notCovered = 0

for i, w in enumerate(explanationFile):

    words = w.split(" ")
    t = words[1] + "_" + words[2]

    if t in clusterMap:

        c = clusterMap[t]
        #print(t, c)

        # check if str(c) is in clusterLabel
        cLabel = clusterLabel[str(c)]
        # print (cLabel, words[0])

        if (words[0] == cLabel):
            hit = hit + 1

        total = total + 1


    else:
        notCovered = notCovered + 1

accuracy = (hit / total) * 100
print("Accuracy, hits, total, words not covered")
print(accuracy, hit, total, notCovered)

