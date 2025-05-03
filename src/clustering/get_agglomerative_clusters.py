import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn import manifold
import time
import codecs
import argparse
import dill as pickle
import os
import sys

def check_files_exist(files):
	for f in files:
		if not os.path.exists(f):
			print(f"Error: Required file {f} does not exist!")
			return False
	return True

def save_clusters(clusters, vocab, labels, outputpath, K):
	# Save clustering labels
	fn = outputpath + '/labels-' + str(K) + '.txt'
	try:
		with open(fn, 'w') as f:
			for label in labels:
				f.write(str(label) + '\n')
	except IOError as e:
		print(f"Error writing labels file: {e}")
		return False

	# Save clusters
	try:
		with open(outputpath+'/clusters-'+str(K)+'.txt', 'w') as target:
			for key in clusters.keys():
				for word in clusters[key]:
					target.write(word + "|||" + str(key) + "\n")
	except IOError as e:
		print(f"Error writing clusters file: {e}")
		return False
	
	return True

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--vocab-file","-v", required=True, help="output vocab file with complete path")
	parser.add_argument("--point-file","-p", required=True, help="output point file with complete path")
	parser.add_argument("--output-path","-o", required=True, help="output path clustering model and result files")
	parser.add_argument("--cluster","-k", required=True, help="cluster numbers comma separated (e.g. 5,10,15)")
	parser.add_argument("--range","-r", type=bool, default=False, 
					   help="whether cluster option provides a range or cluster numbers (e.g. in case of range: 5,50,10 start with k=5 increment by 10 till reach k=50)")
	parser.add_argument("--batch-size", type=int, default=1024, 
					   help="batch size for MiniBatchKMeans (default: 1024)")

	args = parser.parse_args()

	# Check if required files exist
	if not check_files_exist([args.vocab_file, args.point_file]):
		sys.exit(1)

	# Create output directory if it doesn't exist
	os.makedirs(args.output_path, exist_ok=True)

	# Process cluster sizes
	try:
		Ks = [int(k) for k in args.cluster.split(',')]
		if args.range:
			Ks = list(range(Ks[0], Ks[1]+1, Ks[2]))
		print("Will process these cluster sizes:", Ks)
	except ValueError as e:
		print(f"Error processing cluster sizes: {e}")
		sys.exit(1)

	# Load data
	try:
		print("Loading vocabulary...")
		vocab = np.load(args.vocab_file, allow_pickle=True)
		print("Loading points...")
		points = np.load(args.point_file, allow_pickle=True)
	except Exception as e:
		print(f"Error loading data files: {e}")
		sys.exit(1)

	# Process each K
	for K in Ks:
		starttime = time.time()
		print(f"Performing {K}-means clustering...")
		
		try:
			# Use MiniBatchKMeans for memory efficiency
			clustering = MiniBatchKMeans(n_clusters=K, 
									   batch_size=args.batch_size,
									   random_state=0)
			clustering.fit(points)

			# Save model
			model_file = f"{args.output_path}/model-{K}-kmeans-clustering.pkl"
			with open(model_file, "wb") as fp:
				pickle.dump(clustering, fp)

			# Process clusters
			clusters = defaultdict(list)
			for i, label in enumerate(clustering.labels_):
				clusters[label].append(vocab[i])

			# Save results
			if not save_clusters(clusters, vocab, clustering.labels_, args.output_path, K):
				print(f"Error saving results for K={K}")
				continue

			endtime = time.time()
			print(f"K={K}: Time taken: {endtime - starttime:.2f} sec")

		except Exception as e:
			print(f"Error processing K={K}: {e}")
			continue

if __name__ == "__main__":
	main()
