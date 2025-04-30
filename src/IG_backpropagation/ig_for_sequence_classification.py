import argparse
import warnings

from abc import ABC, abstractmethod

import pandas as pd
from captum.attr import (
    LayerIntegratedGradients,
    LayerGradientXActivation,
    LayerDeepLift,
)

import numpy as np

import torch

from transformers import BertTokenizer, BertForTokenClassification, RobertaTokenizer, RobertaForTokenClassification

warnings.filterwarnings("ignore", message="Detokenization Failed")

class Explainer(ABC):
    def __init__(self, model, tokenizer, no_detokenize=False, device=None):
        # Autodetect if device is None
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.no_detokenize = no_detokenize

    @abstractmethod
    def init_explainer(self, *args, **kwargs):
        pass

    @abstractmethod
    def interpret(self, sentence, *args, **kwargs):
        pass

    def _detokenize_explanation(self, sentence, tokenized_explanation, method="max"):
        assert method in ["max", "avg", "first", "last"]

        detokenized_explanation = []
        line = sentence.strip()
        original_tokens = line.split(" ")

        current_idx = 0
        while current_idx < len(tokenized_explanation):
            # Handle special tokens
            if tokenized_explanation[current_idx][0] in self.tokenizer.all_special_tokens:
                detokenized_explanation.append(tokenized_explanation[current_idx])
                current_idx += 1
                continue

            # Get current token and clean it
            current_token = tokenized_explanation[current_idx][0]
            if current_token.startswith('Ġ'):
                current_token = current_token[1:]

            # More flexible matching
            found_match = False
            for orig_token in original_tokens:
                if (orig_token == current_token or 
                    orig_token.lower() == current_token.lower() or
                    orig_token.startswith(current_token) or 
                    current_token.startswith(orig_token)):
                    
                    # Calculate how many subwords this token might have
                    remaining_tokens = tokenized_explanation[current_idx:]
                    total_length = 1  # At least include current token
                    for j, (token, _) in enumerate(remaining_tokens[1:], 1):  # Start from next token
                        if token.startswith('Ġ') or (current_idx + j) >= len(tokenized_explanation):
                            break
                        total_length += 1

                    # Safety check for total_length
                    total_length = min(total_length, len(tokenized_explanation) - current_idx)
                    
                    if total_length > 0:
                        # Get the relevant tokens
                        relevant_tokens = tokenized_explanation[current_idx:current_idx + total_length]
                        
                        # Aggregate the attributions based on method
                        if method == "first":
                            value = relevant_tokens[0][1]
                        elif method == "last":
                            value = relevant_tokens[-1][1]
                        elif method == "max":
                            value = max(t[1] for t in relevant_tokens)
                        else:  # avg
                            value = sum(t[1] for t in relevant_tokens) / len(relevant_tokens)

                        detokenized_explanation.append((orig_token, value))
                        current_idx += total_length
                        found_match = True
                        break
                    else:
                        # Fallback if no valid length found
                        detokenized_explanation.append((orig_token, tokenized_explanation[current_idx][1]))
                        current_idx += 1
                        found_match = True
                        break

            if not found_match:
                # If no match found, just use the token as is with its attribution
                detokenized_explanation.append((tokenized_explanation[current_idx][0], tokenized_explanation[current_idx][1]))
                current_idx += 1

        return detokenized_explanation


class IGExplainer(Explainer):
    def init_explainer(self, layer=0, *args, **kwargs):
        self.custom_forward = lambda *inputs: self.model(*inputs).logits

        # Check if it's a RoBERTa or BERT model
        if hasattr(self.model, 'roberta'):
            base_model = self.model.roberta
        else:
            base_model = self.model.bert

        # Layer 0 is embedding
        if layer == 0:
            self.interpreter = LayerIntegratedGradients(
                self.custom_forward, base_model.embeddings
            )
        else:
            self.interpreter = LayerIntegratedGradients(
                self.custom_forward, base_model.encoder.layer[int(layer) - 1]
            )

    def _summarize_attributions(self, attributions):
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        return attributions

    def interpret(self, sentence, *args, **kwargs):
        # Add truncation=True to handle long sequences
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)

        inputs = inputs.to(self.device)

        logits = self.custom_forward(inputs["input_ids"], inputs["attention_mask"])
        logits = logits[0, :].detach().squeeze()
        predicted_class_idx = np.argmax(logits.cpu().numpy())
        predicted_class = self.model.config.id2label[predicted_class_idx]
        predicted_confidence = round(
            torch.softmax(logits, dim=-1)[predicted_class_idx].item(), 2
        )

        interpreter_args = {
            "baselines": kwargs.get("baselines", None),
            "additional_forward_args": (inputs["attention_mask"],),
            "target": (predicted_class_idx,),
            "n_steps": kwargs.get("n_steps", 500),
            "return_convergence_delta": True,
            "internal_batch_size": 10000 // inputs["input_ids"].shape[1],
        }

        attributions, delta = self.interpreter.attribute(
            inputs["input_ids"], **interpreter_args
        )

        input_saliencies = self._summarize_attributions(attributions).tolist()

        tokenized_explanation = list(
            zip(
                self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]),
                input_saliencies,
            )
        )

        if self.no_detokenize:
            explanations = {"Raw": tokenized_explanation}
        else:
            explanations = {
                "Raw": tokenized_explanation,
                "Maximum of subtokens": self._detokenize_explanation(
                    sentence, tokenized_explanation, method="max"
                ),
                "Average of subtokens": self._detokenize_explanation(
                    sentence, tokenized_explanation, method="avg"
                ),
                "First Subtoken": self._detokenize_explanation(
                    sentence, tokenized_explanation, method="first"
                ),
                "Last Subtoken": self._detokenize_explanation(
                    sentence, tokenized_explanation, method="last"
                ),
            }

        return (sentence, predicted_class, predicted_confidence, explanations)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--save_file", required=True)
    parser.add_argument("--model_type", type=str, default="roberta", 
                       choices=["bert", "roberta"], 
                       help="Type of model to use")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use the correct model type based on argument
    if args.model_type == "roberta":
        model = RobertaForSequenceClassification.from_pretrained(args.model).to(device)
        tokenizer = RobertaTokenizer.from_pretrained(args.model)
    else:
        model = BertForTokenClassification.from_pretrained(args.model).to(device)
        tokenizer = BertTokenizer.from_pretrained(args.model)

    explainer = IGExplainer(model, tokenizer, device=device)
    explainer.init_explainer(layer=args.layer)

    # create a pandas dataframe to store the results
    df = pd.DataFrame(columns=["sentence_id", "predicted_class", "predicted_confidence", "saliencies"])


    senten_idx = []
    prediction = []
    confidence = []
    all_saliencies = []
    with open(args.input_file) as fp:
        for sentence_idx, line in enumerate(fp):
            try:
                result = explainer.interpret(line.strip())
                sentence, predicted_class, predicted_confidence, explanations = result
                print(f"Sentence: {sentence}")
                print(
                    f"Predicted class: {predicted_class} ({predicted_confidence*100:.2f}%)"
                )
                saliencies = explanations["Maximum of subtokens"]
                saliencies = [(w, abs(s)) for w, s in saliencies]
                print(f"Saliencies: {saliencies}")

                senten_idx.append(sentence_idx)
                prediction.append(predicted_class)
                confidence.append(predicted_confidence)
                all_saliencies.append(saliencies)
            except RuntimeError as e:
                print(f"Skipping line {sentence_idx} due to error: {str(e)}")
                continue

    df["sentence_id"] = senten_idx
    df["predicted_class"] = prediction
    df["predicted_confidence"] = confidence
    df["saliencies"] = all_saliencies

    df.to_csv(args.save_file, index=False)

if __name__ == "__main__":
    main()