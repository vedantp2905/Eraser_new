import json
from argparse import ArgumentParser
import pandas as pd
import time
from tqdm import tqdm


def fileRead(fname, flag):
    lines = []
    if flag == 0:
        with open(fname, "r") as f:
            data = json.load(f)
            lines = [line.rstrip('\r\n') for line in data]
    else:
        print("Reading file: " + fname)
        with open(fname) as f:
            lines = [line.rstrip('\r\n') for line in f]
    return lines


# read the json file [text.in.tok.sent_len_min_0_max_1000000_del_1000000-dataset.json] (token & token representation)
def read_dataset_file(fname):
    print("Reading dataset file...")
    dataset_map = {}
    with open(fname, 'r') as json_file:
        json_data = json.load(json_file)
        
        for value in json_data:
            parts = value[0].split("|||")
            token = parts[0]
            sentence_idx = parts[2]
            word_idx = parts[3]
            token_rep = value[1]
            
            dataset_map[f"{word_idx}_{sentence_idx}"] = (token, token_rep)
    
    print(f"Loaded {len(dataset_map)} entries from dataset")
    return dataset_map


def match_to_representation(explanationFile, dataset_map):
    print("Matching representations...")
    matched_data = []
    skipped = 0
    
    for i, w in enumerate(tqdm(explanationFile)):
        try:
            parts = w.split()
            if len(parts) != 3:
                print(f"Warning: Malformed line {i}: {w}")
                skipped += 1
                continue
                
            label, word_idx, sentence_idx = parts
            key = f"{word_idx}_{sentence_idx}"
            
            if key not in dataset_map:
                print(f"Warning: Key not found in dataset: {key}")
                skipped += 1
                continue
                
            token, token_rep = dataset_map[key]
            matched_data.append({
                'sentence_idx': sentence_idx,
                'word_idx': word_idx,
                'token': token,
                'token_rep': token_rep,
                'label': label
            })
            
        except Exception as e:
            print(f"Error processing line {i}: {w}")
            print(f"Error details: {str(e)}")
            skipped += 1
            continue
    
    print(f"Processed {len(explanationFile)} entries, skipped {skipped}")
    
    if not matched_data:
        raise ValueError("No valid entries found after matching!")
    
    # Unzip the matched data into separate lists
    result = {
        'sentence_idx': [],
        'word_idx': [],
        'tokens': [],
        'token_rep': [],
        'labels': []
    }
    
    for entry in matched_data:
        result['sentence_idx'].append(entry['sentence_idx'])
        result['word_idx'].append(entry['word_idx'])
        result['tokens'].append(entry['token'])
        result['token_rep'].append(entry['token_rep'])
        result['labels'].append(entry['label'])
    
    return (result['sentence_idx'], result['word_idx'], 
            result['tokens'], result['token_rep'], result['labels'])


def generate_csv_file(all_sentence_idx, all_word_idx, all_tokens, all_token_rep, all_labels, output_file):
    print("Generating CSV file...")
    # Verify all arrays are the same length
    lengths = [len(x) for x in [all_sentence_idx, all_word_idx, all_tokens, all_token_rep, all_labels]]
    if len(set(lengths)) != 1:
        raise ValueError(f"Arrays have different lengths: {lengths}")
        
    df = pd.DataFrame({
        'token': all_tokens,
        'line_idx': all_sentence_idx,
        'position_idx': all_word_idx,
        'embedding': all_token_rep,
        'labels': all_labels
    })
    
    print(f"Writing {len(df)} entries to {output_file}")
    df.to_csv(output_file, index=False, sep='\t')


def main():
    parser = ArgumentParser()
    parser.add_argument("--explanationFile", type=str, required=True,
                       help="Path to the explanation file")
    parser.add_argument("--datasetFile", type=str, required=True,
                       help="Path to the dataset file")
    parser.add_argument("--outputFile", type=str, required=True,
                       help="Path to the output file")
    args = parser.parse_args()

    try:
        start_time = time.time()
        
        # Load files
        explanationFile = fileRead(args.explanationFile, 1)
        dataset_map = read_dataset_file(args.datasetFile)
        
        # Process data
        result = match_to_representation(explanationFile, dataset_map)
        
        # Generate output
        generate_csv_file(*result, args.outputFile)
        
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()










