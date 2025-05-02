""" Generate the explanation file for [CLS] tokens.

This scripts will get the predicted label for each sentence in the dataset, and then use the predicted label,
sentence_idx and word_idx to generate the explanation file for CLS tokens.
"""

import argparse
import pandas as pd
from transformers import AutoTokenizer, RobertaForTokenClassification
import torch
import torch.nn.functional as F
import os
import json
from tqdm import tqdm


def get_dataset(file_name):
    with open(file_name, 'r') as f:
        data = f.readlines()
    return [line.strip() for line in data]


def process_batch(model, tokenizer, sentences, device, batch_size=32):
    """Process sentences in batches to avoid memory issues"""
    all_labels = []
    all_input_ids = []
    
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(batch_sentences, 
                         padding=True, 
                         truncation=True, 
                         return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = F.softmax(outputs[0], dim=-1)
            batch_labels = torch.argmax(predictions, axis=1).tolist()
            
        all_labels.extend(batch_labels)
        all_input_ids.extend(inputs['input_ids'].cpu().tolist())
        
        # Clear CUDA cache after each batch
        if device == "cuda":
            torch.cuda.empty_cache()
    
    return all_input_ids, all_labels


def save_prediction(labels, sentences, output_path):
    path = output_path + '/predicted.csv'
    df = pd.DataFrame({'sentence': sentences, 'labels': labels})
    df.to_csv(path, sep='\t', index=False)


def generate_explanation(ids, labels, save_path):
    """
    Generate and save the explanation for the [CLS] tokens.

    Parameters
    ----------
    ids: torch.Tensor
        The token ids of the sentences.

    labels: List
        The predicted labels of the sentences.

    save_path: str
        The path to save the json file.

    Returns
    -------
    layer_cls_info: list
        The [CLS] token representation and info.
    """
    predictions = []
    
    for j in range(len(ids)):
        predicted_result = labels[j]
        predictions.append(f"{predicted_result} -1 {j}")

    os.makedirs(save_path, exist_ok=True)
    
    path = save_path + '/explanation_CLS.txt'
    with open(path, "w") as txt_file:
        for line in predictions:
            txt_file.write(line + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name-or-path', type=str, required=True,
                       help='The name or path of the dataset to be loaded.')
    parser.add_argument('--model-name', type=str, required=True,
                       help='The name or path of the pre-trained model to be used.')
    parser.add_argument('--tokenizer-name', type=str, required=True,
                       help='The name or path of the tokenizer to be used.')
    parser.add_argument('--save-dir', type=str, required=True,
                       help='The directory to save the extracted [CLS] token representation and info.')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for processing (default: 32)')
    parser.add_argument('--cpu-only', action='store_true',
                       help='Force CPU usage even if CUDA is available')

    args = parser.parse_args()

    # Load dataset
    print("Loading dataset...")
    sentences = get_dataset(args.dataset_name_or_path)
    
    # Setup device
    device = "cpu" if args.cpu_only else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = RobertaForTokenClassification.from_pretrained(args.model_name).to(device)
    model.eval()  # Set model to evaluation mode

    print(f"Processing {len(sentences)} sentences in batches of {args.batch_size}...")
    try:
        input_ids, labels = process_batch(model, tokenizer, sentences, device, args.batch_size)
        
        print("Saving predictions...")
        save_prediction(labels, sentences, args.save_dir)
        
        print("Generating explanations...")
        generate_explanation(input_ids, labels, args.save_dir)
        
        print("Successfully completed all operations!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


if __name__ == '__main__':
    main()

