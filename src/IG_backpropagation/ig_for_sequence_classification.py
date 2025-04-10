import argparse

from abc import ABC, abstractmethod

import pandas as pd
from captum.attr import (
    LayerIntegratedGradients,
    LayerGradientXActivation,
    LayerDeepLift,
)

import numpy as np

import torch

from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification


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
        for token in original_tokens:
            # Skip if we've reached the end of tokenized explanations
            if current_idx >= len(tokenized_explanation):
                break
            
            # Skip special tokens
            while (current_idx < len(tokenized_explanation) and 
                   tokenized_explanation[current_idx][0] in self.tokenizer.all_special_tokens):
                detokenized_explanation.append(tokenized_explanation[current_idx])
                current_idx += 1
            
            if current_idx >= len(tokenized_explanation):
                break

            # Remove the 'Ġ' prefix for comparison
            current_token = tokenized_explanation[current_idx][0]
            if current_token.startswith('Ġ'):
                current_token = current_token[1:]

            # More flexible matching
            if not (token.lower() == current_token.lower() or 
                    token.lower().startswith(current_token.lower()) or 
                    current_token.lower().startswith(token.lower())):
                print(f"[WARNING] Detokenization Failed at {token} vs {tokenized_explanation[current_idx][0]}")
            
            tokenized_length = max(1, len(self.tokenizer.tokenize(token)))
            
            try:
                if method == "first":
                    detokenized_explanation.append((token, tokenized_explanation[current_idx][1]))
                elif method == "last":
                    detokenized_explanation.append((token, tokenized_explanation[min(current_idx + tokenized_length - 1, 
                                                                                  len(tokenized_explanation) - 1)][1]))
                elif method == "max":
                    max_attrib = max([tokenized_explanation[idx][1] 
                                    for idx in range(current_idx, 
                                                   min(current_idx + tokenized_length,
                                                       len(tokenized_explanation)))])
                    detokenized_explanation.append((token, max_attrib))
                elif method == "avg":
                    avg_attrib = sum([tokenized_explanation[idx][1] 
                                    for idx in range(current_idx,
                                                  min(current_idx + tokenized_length,
                                                      len(tokenized_explanation)))]) / tokenized_length
                    detokenized_explanation.append((token, avg_attrib))
            except Exception as e:
                print(f"[WARNING] Error processing token {token}: {str(e)}")
                detokenized_explanation.append((token, 0.0))  # fallback value
            
            current_idx += tokenized_length

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
        inputs = self.tokenizer(sentence, return_tensors="pt")

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
        model = BertForSequenceClassification.from_pretrained(args.model).to(device)
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
            result = explainer.interpret(line.strip())

            sentence, predicted_class, predicted_confidence, explanations = result
            print(f"Sentence: {sentence}")
            print(
                f"Predicted class: {predicted_class} ({predicted_confidence*100:.2f}%)"
            )
            saliencies = explanations["Maximum of subtokens"]
            saliencies = [(w, abs(s)) for w, s in saliencies]
            print(f"Saliencies: {saliencies}")

            # label for 0 if the predicted class is negative, 1 if positive
            # label = 0 if predicted_class == "negative" else 1

            senten_idx.append(sentence_idx)
            prediction.append(predicted_class)
            confidence.append(predicted_confidence)
            all_saliencies.append(saliencies)

    df["sentence_id"] = senten_idx
    df["predicted_class"] = prediction
    df["predicted_confidence"] = confidence
    df["saliencies"] = all_saliencies

    df.to_csv(args.save_file, index=False)

if __name__ == "__main__":
    main()
