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

from transformers import BertTokenizer, BertForSequenceClassification


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

        idx_to_pick = []
        current_idx = 0
        for token in original_tokens:
            while (
                tokenized_explanation[current_idx][0]
                in self.tokenizer.all_special_tokens
            ):
                detokenized_explanation.append(tokenized_explanation[current_idx])
                current_idx += 1

            if not token.startswith(
                tokenized_explanation[current_idx][0]
            ) and not token.lower().startswith(tokenized_explanation[current_idx][0]):
                print(
                    f"[WARNING] Detokenization Failed at {token} vs {tokenized_explanation[current_idx][0]}"
                )
            tokenized_length = len(self.tokenizer.tokenize(token))
            if method == "first":
                detokenized_explanation.append(
                    (token, tokenized_explanation[current_idx][1])
                )
                current_idx += tokenized_length
            elif method == "last":
                current_idx += tokenized_length
                detokenized_explanation.append(
                    (token, tokenized_explanation[current_idx - 1][1])
                )
            elif method == "max":
                start_idx = current_idx
                current_idx += tokenized_length
                max_attrib = max(
                    [
                        tokenized_explanation[idx][1]
                        for idx in range(start_idx, current_idx)
                    ]
                )
                detokenized_explanation.append((token, max_attrib))
            elif method == "avg":
                start_idx = current_idx
                current_idx += tokenized_length
                avg_attrib = sum(
                    [
                        tokenized_explanation[idx][1]
                        for idx in range(start_idx, current_idx)
                    ]
                ) / (current_idx - start_idx)
                detokenized_explanation.append((token, avg_attrib))

        while (
            current_idx < len(tokenized_explanation)
            and tokenized_explanation[current_idx][0]
            in self.tokenizer.all_special_tokens
        ):
            detokenized_explanation.append(tokenized_explanation[current_idx])
            current_idx += 1

        return detokenized_explanation


class IGRegressionExplainer(Explainer):
    def init_explainer(self, layer=0, *args, **kwargs):
        self.custom_forward = lambda *inputs: self.model(*inputs).logits

        # Layer 0 is embedding
        if layer == 0:
            self.interpreter = LayerIntegratedGradients(
                self.custom_forward, self.model.bert.embeddings
            )
        else:
            self.interpreter = LayerIntegratedGradients(
                self.custom_forward, self.model.bert.encoder.layer[int(layer) - 1]
            )

    def _summarize_attributions(self, attributions):
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        return attributions

    def interpret(self, sentences, *args, **kwargs):
        sentence = f"{sentences[0]} {sentences[1]}"
        inputs = self.tokenizer(*sentences, return_tensors="pt")

        inputs = inputs.to(self.device)

        logits = self.custom_forward(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"])
        logits = logits[0, :].detach().squeeze()
        prediction = logits

        interpreter_args = {
            "baselines": kwargs.get("baselines", None),
            "additional_forward_args": (inputs["attention_mask"],),
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

        return (sentence, prediction, -1, explanations)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file")
    parser.add_argument("model")
    parser.add_argument("layer", type=int)
    parser.add_argument("save_file")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(args.model).to(device)
    tokenizer = BertTokenizer.from_pretrained(args.model)

    explainer = IGRegressionExplainer(model, tokenizer)

    explainer.init_explainer(layer=args.layer)

    # create a pandas dataframe to store the results
    df = pd.DataFrame(columns=["sentence_id", "predicted_class", "predicted_confidence", "saliencies"])

    sentence_idx_list = []
    predicted_class_list = []
    predicted_confidence_list = []
    saliencies_list = []

    with open(args.input_file) as fp:
        for sentence_idx, line in enumerate(fp):
            sentence = line.strip().split("\t")
            sentence_1 = sentence[0].strip()
            sentence_2 = sentence[1].strip() # CHANGE THIS

            result = explainer.interpret((sentence_1, sentence_2))

            sentence, prediction, predicted_confidence, explanations = result
            print(f"Sentence: {sentence}")
            print(
                f"Prediction: {prediction} ({predicted_confidence*100:.2f}%)"
            )
            saliencies = explanations["Maximum of subtokens"]
            saliencies = [(w, abs(s)) for w, s in saliencies]
            print(f"Saliencies: {saliencies}")

            sentence_idx_list.append(sentence_idx)
            predicted_class_list.append(prediction)
            predicted_confidence_list.append(predicted_confidence)
            saliencies_list.append(saliencies)

    # convert the score to a value between 0 and 5
    # Define the minimum and maximum values of the predicted scores
    min_pred = min(predicted_class_list)
    max_pred = max(predicted_class_list)

    # Define the desired range of the labels
    min_label = 0
    max_label = 5

    # Scale the predicted scores to the range of 0 to 5
    predicted_score = [(score - min_pred) / (max_pred - min_pred) * (max_label - min_label) + min_label for score in predicted_class_list]

    label = []
    for score in predicted_score:
        if score < 2.5:
            label.append(0)
        else:
            label.append(1)

    df["sentence_id"] = sentence_idx_list
    df["predicted_class"] = label
    df["predicted_confidence"] = predicted_confidence_list
    df["saliencies"] = saliencies_list

    df.to_csv(args.save_file, index=False)


if __name__ == "__main__":
    main()
