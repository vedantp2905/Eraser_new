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

from transformers import AutoTokenizer, AutoModelForSequenceClassification


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

    def _detokenize_explanation(self, tokenized_explanation):
        """Detokenizes subword tokens and aggregates their attributions."""
        current_token = ""
        current_attribution = 0.0
        detokenized_explanation = []
        
        current_idx = 0
        while current_idx < len(tokenized_explanation):
            token, attribution = tokenized_explanation[current_idx]
            
            # Handle RoBERTa's special "Ġ" prefix
            if token.startswith('Ġ'):
                token = token[1:]  # Remove the Ġ prefix
                if current_token:  # Save previous token if exists
                    detokenized_explanation.append((current_token, current_attribution))
                current_token = token
                current_attribution = attribution
            else:
                # For continuation tokens, append without space
                current_token += token
                current_attribution = max(current_attribution, attribution)
            
            current_idx += 1
        
        # Add the last token
        if current_token:
            detokenized_explanation.append((current_token, current_attribution))
        
        return detokenized_explanation


class IGExplainer(Explainer):
    def init_explainer(self, layer=0, *args, **kwargs):
        self.custom_forward = lambda *inputs: self.model(*inputs).logits

        # Layer 0 is embedding
        if layer == 0:
            self.interpreter = LayerIntegratedGradients(
                self.custom_forward, self.model.roberta.embeddings
            )
        else:
            self.interpreter = LayerIntegratedGradients(
                self.custom_forward, self.model.roberta.encoder.layer[int(layer) - 1]
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
                    tokenized_explanation
                ),
            }

        return (sentence, predicted_class, predicted_confidence, explanations)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file")
    parser.add_argument("model")
    parser.add_argument("layer", type=int)
    parser.add_argument("save_file")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

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
