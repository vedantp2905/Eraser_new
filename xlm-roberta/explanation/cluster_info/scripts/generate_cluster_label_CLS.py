#!/usr/bin/env python
import sys
import json
import operator
from collections import Counter
from argparse import ArgumentParser


def clusterRead(fname):
    words = []
    words_idx = []
    cluster_idx = []
    sent_idx = []

    with open(fname) as f:
        for line in f:
            line = line.rstrip('\r\n')
            parts = line.split("|||")
            
            if int(parts[3]) == -1:
                words.append(parts[0])
                cluster_idx.append(int(parts[4]))
                words_idx.append(int(parts[3]))
                sent_idx.append(int(parts[2]))

    return words, words_idx, sent_idx, cluster_idx


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


parser = ArgumentParser()
parser.add_argument("-c", "--clusterFile", dest="c",
                    help="Cluster File", metavar="FILE")
parser.add_argument("-e", "--explanationFile", dest="e",
                    help="Explanation File", metavar="FILE")
parser.add_argument("-p", "--percentage", dest="p",
                    help="Percentage", metavar="FILE")
parser.add_argument("-s", "--save_path", dest="s",
                    help="path of save directory", metavar="FILE")                   

args = parser.parse_args()

explanationMap = {}

explanationFile = fileRead(args.e, 1)

for i, w in enumerate(explanationFile):
    words = w.split(" ")

    t = words[1] + "_" + words[2]
    explanationMap[t] = words[0]

words, words_idx, sent_idx, cluster_idx = clusterRead(args.c)

labelCount = {}
clusterLabel = {}
clusterMap = {}
prev = cluster_idx[0]
aligned = 0
threshold = args.p

for i, w in enumerate(cluster_idx):

    if (prev == cluster_idx[i]):

        thisEntry = str(words_idx[i]) + "_" + str(sent_idx[i])
        thisLabel = explanationMap[thisEntry]
        # print (thisLabel)

        if (thisLabel in labelCount):
            labelCount[thisLabel] = labelCount[thisLabel] + 1
        else:
            labelCount[thisLabel] = 1

        prev = cluster_idx[i]

    else:

        cluster = dict(sorted(labelCount.items(), key=lambda item: item[1], reverse=True))
        sum = 0
        highest = list(cluster.values())[0]

        for k, v in labelCount.items():
            sum = sum + v

        if ((highest / sum * 100) >= float(threshold)):
            # print ("Cluster " + str(cluster_idx[i-1]) + " is " + list(cluster.keys())[0])
            # print (list(cluster.keys())[0])
            clusterLabel[str(cluster_idx[i - 1])] = list(cluster.keys())[0]
            aligned = aligned + 1
        else:
            clusterLabel[str(cluster_idx[i - 1])] = "Mix Label"
        # print ("Cluster not found", cluster)

        labelCount = {}

        prev = cluster_idx[i]
        thisEntry = str(words_idx[i]) + "_" + str(sent_idx[i])
        thisLabel = explanationMap[thisEntry]
        labelCount[thisLabel] = 1

    clusterMap[thisEntry] = cluster_idx[i]

cluster = dict(sorted(labelCount.items(), key=lambda item: item[1], reverse=True))
sum = 0
highest = list(cluster.values())[0]

for k, v in labelCount.items():
    sum = sum + v

if ((highest / sum * 100) >= float(threshold)):
    # print ("Cluster " + str(cluster_idx[i-1]) + " is " + list(cluster.keys())[0])
    # print (list(cluster.keys())[0])
    clusterLabel[str(cluster_idx[i - 1])] = list(cluster.keys())[0]
    aligned = aligned + 1
else:
    clusterLabel[str(cluster_idx[i - 1])] = "Mix Label"

print("Aligned", aligned)

# go through clusterLabel and save the clusters within the same label
clusterMap = {}
for k, v in clusterLabel.items():
    if v not in clusterMap:
        clusterMap[v] = []
    clusterMap[v].append(k)


# save the clusterLabel
with open(args.s, 'w') as fp:
    json.dump(clusterLabel, fp, indent=4)










