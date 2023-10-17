# read csv file by pandas
import argparse
import json


def clusterRead(fname, is_CLS_only):
    words = []
    words_idx = []
    cluster_idx = []
    sent_idx = []

    with open(fname) as f:
        for line in f:
            line = line.rstrip('\r\n')
            parts = line.split("|||")

            if is_CLS_only:
                if int(parts[3]) == -1:
                    words.append(parts[0])
                    cluster_idx.append(int(parts[4]))
                    words_idx.append(int(parts[3]))
                    sent_idx.append(int(parts[2]))
            else:
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

def check_cluster_contain_CLS(words_idx, cluster_idx):
    # print the cluster id for the token has the position_idx -1
    CLS_cluster_id = []
    for i in range(len(words_idx)):
        if words_idx[i] == -1:
            CLS_cluster_id.append(cluster_idx[i])

    # remove the duplicate cluster id
    CLS_cluster_id = list(set(CLS_cluster_id))

    return CLS_cluster_id


def check_CLS_dominant_cluster(CLS_cluster_id, words, words_idx, cluster_idx, threshold):
    # count the number of CLS token in each cluster
    CLS_dominant_cluster = []
    for i in range(len(CLS_cluster_id)):
        count = 0
        total = 0
        for j in range(len(words)):
            if cluster_idx[j] == CLS_cluster_id[i]:
                if words_idx[j] == -1:
                    count += 1
                total += 1
        if (count / total * 100) >= float(threshold):
            CLS_dominant_cluster.append(CLS_cluster_id[i])

    return CLS_dominant_cluster


def get_dominate_cluster(CLS_dominant_cluster, cluster_idx, words_idx, sent_idx, words):
    subset_cluster_idx = []
    subset_words_idx = []
    subset_words = []
    subset_sent_idx = []

    print(len(cluster_idx), len(words_idx), len(sent_idx), len(words))
    
    for i in range(len(CLS_dominant_cluster)):
        for j in range(len(cluster_idx)):
            if cluster_idx[j] == CLS_dominant_cluster[i]:
                subset_cluster_idx.append(cluster_idx[j])
                subset_words_idx.append(words_idx[j])
                subset_words.append(words[j])
                subset_sent_idx.append(sent_idx[j])
                
    return subset_cluster_idx, subset_words_idx, subset_words, subset_sent_idx


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("-c", "--clusterFile", dest="c", help="Cluster File", metavar="FILE")
    parse.add_argument("-e", "--explanationFile", dest="e", help="Explanation File", metavar="FILE")
    parse.add_argument("-s", "--save_path", dest="s", help="path of save directory", metavar="FILE")  
    parse.add_argument("-p", "--percentage", dest="p", help="Percentage", metavar="FILE")
    args = parse.parse_args()

    words_list, words_idx, sent_idx, cluster_idx = clusterRead(args.c, False)
    CLS_cluster_id = check_cluster_contain_CLS(words_idx, cluster_idx)
    CLS_dominant_cluster = check_CLS_dominant_cluster(CLS_cluster_id, words_list, words_idx, cluster_idx, args.p)

    explanationMap = {}

    explanationFile = fileRead(args.e, 1)

    for i, w in enumerate(explanationFile):
        words = w.split(" ")

        t = words[1] + "_" + words[2]
        explanationMap[t] = words[0]

    if len(CLS_dominant_cluster) > 0:
        words_list, words_idx, sent_idx, cluster_idx = clusterRead(args.c, True)
        cluster_idx, words_idx, words_list, sent_idx = get_dominate_cluster(CLS_dominant_cluster, cluster_idx, words_idx, sent_idx, words_list)

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
    else:
        print("No CLS dominant clusters")


if __name__ == '__main__':
    main()
