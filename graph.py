import os
import pandas as pd
import matplotlib.pyplot as plt
import ast  # For safely evaluating strings as Python objects

# Directory containing the CSV files
outDir = "eraser_movie/IG_attributions"

# Initialize a dictionary to store saliencies for each layer
saliencies = {layer: [] for layer in range(13)}

# Load the CSV files for each layer
for layer in range(13):
    file_path = os.path.join(outDir, f"IG_explanation_layer_{layer}.csv")
    df = pd.read_csv(file_path)
    
    # Extract saliencies for each sentence
    for _, row in df.iterrows():
        try:
            # Safely evaluate the 'saliencies' string as a list of tuples
            saliency_list = ast.literal_eval(row['saliencies'])
            saliencies[layer].append(saliency_list)
        except (ValueError, SyntaxError):
            print(f"Error parsing saliencies for layer {layer}, sentence_id: {row['sentence_id']}")
            saliencies[layer].append([])  # Append an empty list if parsing fails

# Plot saliencies for each token across layers
for sentence_idx in range(len(saliencies[0])):  # Use the number of sentences from the first layer
    plt.figure(figsize=(12, 6))
    
    # Get the tokens and their saliencies from the first layer (assumed to be the same across all layers)
    tokens = [token for token, _ in saliencies[0][sentence_idx]]
    
    # Calculate average saliency for each token across layers
    token_avg_saliencies = []
    for token_idx, token in enumerate(tokens):
        try:
            # Extract saliency values for this token across layers
            token_saliencies = [saliencies[layer][sentence_idx][token_idx][1] for layer in range(13)]
            avg_saliency = sum(token_saliencies) / len(token_saliencies)
            token_avg_saliencies.append((token, avg_saliency))
        except IndexError:
            print(f"IndexError: Skipping token {token} in sentence {sentence_idx + 1}")
            continue
    
    # Sort tokens by average saliency and select the top 10
    top_tokens = sorted(token_avg_saliencies, key=lambda x: x[1], reverse=True)[:10]
    
    # Include the [CLS] token if it exists in the tokens list
    cls_token = "<s>"
    if cls_token in tokens:
        cls_avg_saliency = next((avg for token, avg in token_avg_saliencies if token == cls_token), None)
        if cls_avg_saliency is not None:
            top_tokens.append((cls_token, cls_avg_saliency))
    
    # Plot the top 10 tokens and the [CLS] token
    for token, avg_saliency in top_tokens:
        token_saliencies = [saliencies[layer][sentence_idx][tokens.index(token)][1] for layer in range(13)]
        plt.plot(range(13), token_saliencies, label=f"{token} (avg: {avg_saliency:.3f})")
    
    # Get the sentence_id from the first layer's CSV
    sentence_id = df.loc[sentence_idx, 'sentence_id']
    
    plt.title(f"Top 10 Tokens and [CLS] Across Layers for Sentence {sentence_id}")
    plt.xlabel("Layer")
    plt.ylabel("Saliency")
    
    # Increase legend size
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='large')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outDir, f"top10_saliencies_sentence_{sentence_id}.png"))
    plt.close()

print("Plots saved to:", outDir)