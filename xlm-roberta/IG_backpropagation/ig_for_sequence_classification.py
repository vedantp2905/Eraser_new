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

from transformers import AutoTokenizer, XLMRobertaForSequenceClassification


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

    def aggregate_repr(self, state, start_idx, end_idx, aggregation, token):
        """
        Function that aggregates activations/embeddings over a span of subword tokens.
        This function will usually be called once per word. For example, if we had the sentence::

            This is an example

        which is tokenized by BPE into::

            this is an ex @@am @@ple

        The function should be called 4 times::

            aggregate_repr(state, 0, 0, aggregation)
            aggregate_repr(state, 1, 1, aggregation)
            aggregate_repr(state, 2, 2, aggregation)
            aggregate_repr(state, 3, 5, aggregation)

        Returns a zero vector if end is less than start, i.e. the request is to
        aggregate over an empty slice.

        Parameters
        ----------
        state : numpy.ndarray
            Matrix of size [ NUM_LAYERS x NUM_SUBWORD_TOKENS_IN_SENT x LAYER_DIM]
        start : int
            Index of the first subword of the word being processed
        end : int
            Index of the last subword of the word being processed
        aggregation : {'first', 'last', 'average'}
            Aggregation method for combining subword activations

        Returns
        -------
        word_vector : numpy.ndarray
            Matrix of size [NUM_LAYERS x LAYER_DIM]
        """
        if end_idx < start_idx:
            sys.stderr.write(
                "WARNING: An empty slice of tokens was encountered. "
                + "This probably implies a special unicode character or text "
                + "encoding issue in your original data that was dropped by the "
                + "transformer model's tokenizer.\n"
            )
            return None
        if aggregation == "first":
            return (token, state[start_idx][1])
        elif aggregation == "last":
            return (token, state[end_idx][1])
        elif aggregation == "average":
            avg_attrib = sum(
                [
                    state[idx][1]
                    for idx in range(start_idx, end_idx + 1)
                ]
            ) / (end_idx - start_idx)
            return (token, avg_attrib)
        elif aggregation == "max":
            max_attrib = max(
                [
                    state[idx][1]
                    for idx in range(start_idx, end_idx + 1)
                ]
            )
            return (token, max_attrib)

    def _detokenize_explanation(self, sentence, tokenized_explanation, aggregation="max"):
        assert aggregation in ["max", "avg", "first", "last"]
        tokenization_counts = {}
        special_tokens = [
            x for x in self.tokenizer.all_special_tokens if x != self.tokenizer.unk_token
        ]
        special_tokens_ids = self.tokenizer.convert_tokens_to_ids(special_tokens)

        original_tokens = sentence.split(" ")

        # Add letters and spaces around each word since some tokenizers are context sensitive
        tmp_tokens = []
        if len(original_tokens) > 0:
            tmp_tokens.append(f"{original_tokens[0]} a")
        tmp_tokens += [f"a {x} a" for x in original_tokens[1:-1]]
        if len(original_tokens) > 1:
            tmp_tokens.append(f"a {original_tokens[-1]}")

        assert len(original_tokens) == len(
            tmp_tokens
        ), f"Original: {original_tokens}, Temp: {tmp_tokens}"

        with torch.no_grad():
            # Get tokenization counts if not already available
            for token_idx, token in enumerate(tmp_tokens):
                tok_ids = [
                    x for x in self.tokenizer.encode(token) if x not in special_tokens_ids
                ]
                # Ignore the added letter tokens
                if token_idx != 0 and token_idx != len(tmp_tokens) - 1:
                    # Word appearing in the middle of the sentence
                    tok_ids = tok_ids[1:-1]
                elif token_idx == 0:
                    # Word appearing at the beginning
                    tok_ids = tok_ids[:-1]
                else:
                    # Word appearing at the end
                    tok_ids = tok_ids[1:]

                if token in tokenization_counts:
                    assert tokenization_counts[token] == len(
                        tok_ids
                    ), "Got different tokenization for already processed word"
                else:
                    tokenization_counts[token] = len(tok_ids)
            ids = self.tokenizer.encode(sentence, truncation=True)
            input_ids = torch.tensor([ids]).to(self.device)

        print('Sentence         : "%s"' % (sentence))
        print("Original    (%03d): %s" % (len(original_tokens), original_tokens))
        print(
            "Tokenized   (%03d): %s"
            % (
                len(self.tokenizer.convert_ids_to_tokens(ids)),
                self.tokenizer.convert_ids_to_tokens(ids),
            )
        )

        assert len(tokenized_explanation) == len(ids)

        # Handle special tokens
        # filtered_ids will contain all ids if we are extracting with
        #  special tokens, and only normal word/subword ids if we are
        #  extracting without special tokens
        # all_hidden_states will also be filtered at this step to match
        #  the ids in filtered ids
        filtered_ids = ids
        idx_special_tokens = [t_i for t_i, x in enumerate(ids) if x in special_tokens_ids]
        special_token_ids = [ids[t_i] for t_i in idx_special_tokens]

        # Get actual tokens for filtered ids in order to do subword
        #  aggregation
        segmented_tokens = self.tokenizer.convert_ids_to_tokens(filtered_ids)

        # Perform subword aggregation/detokenization
        #  After aggregation, we should have |original_tokens| embeddings,
        #  one for each word. If special tokens are included, then we will
        #  have |original_tokens| + |special_tokens|
        counter = 0
        detokenized = []
        detokenized_explanation = [None] * (len(original_tokens) + len(special_token_ids))
        inputs_truncated = False

        # Keep track of what the previous token was. This is used to detect
        #  special tokens followed/preceeded by dropped tokens, which is an
        #  ambiguous situation for the detokenizer
        prev_token_type = "NONE"

        last_special_token_pointer = 0
        for token_idx, token in enumerate(tmp_tokens):
            # Handle special tokens
            if tokenization_counts[token] != 0:
                if last_special_token_pointer < len(idx_special_tokens):
                    while (
                            last_special_token_pointer < len(idx_special_tokens)
                            and counter == idx_special_tokens[last_special_token_pointer]
                    ):
                        assert prev_token_type != "DROPPED", (
                                "A token dropped by the tokenizer appeared next "
                                + "to a special token. Detokenizer cannot resolve "
                                + f"the ambiguity, please remove '{sentence}' from"
                                + "the dataset, or try a different tokenizer"
                        )
                        prev_token_type = "SPECIAL"
                        detokenized_explanation[len(detokenized)] = tokenized_explanation[counter]
                        detokenized.append(
                            segmented_tokens[idx_special_tokens[last_special_token_pointer]]
                        )
                        last_special_token_pointer += 1
                        counter += 1

            current_word_start_idx = counter
            current_word_end_idx = counter + tokenization_counts[token]

            # Check for truncated hidden states in the case where the
            # original word was actually tokenized
            if (
                    tokenization_counts[token] != 0
                    and current_word_start_idx >= len(tokenized_explanation)
            ) or current_word_end_idx > len(tokenized_explanation):
                detokenized_explanation = detokenized_explanation[
                                          : len(detokenized)
                                            + len(special_token_ids)
                                            - last_special_token_pointer,
                                          ]
                inputs_truncated = True
                break

            if tokenization_counts[token] == 0:
                assert prev_token_type != "SPECIAL", (
                        "A token dropped by the tokenizer appeared next "
                        + "to a special token. Detokenizer cannot resolve "
                        + f"the ambiguity, please remove '{sentence}' from"
                        + "the dataset, or try a different tokenizer"
                )
                prev_token_type = "DROPPED"
            else:
                prev_token_type = "NORMAL"

            detokenized_explanation[len(detokenized)] = self.aggregate_repr(
                tokenized_explanation,
                current_word_start_idx,
                current_word_end_idx - 1,
                aggregation,
                original_tokens[token_idx]
            )
            detokenized.append(
                "".join(segmented_tokens[current_word_start_idx:current_word_end_idx])
            )
            counter += tokenization_counts[token]

        while counter < len(segmented_tokens):
            if last_special_token_pointer >= len(idx_special_tokens):
                break

            if counter == idx_special_tokens[last_special_token_pointer]:
                assert prev_token_type != "DROPPED", (
                        "A token dropped by the tokenizer appeared next "
                        + "to a special token. Detokenizer cannot resolve "
                        + f"the ambiguity, please remove '{sentence}' from"
                        + "the dataset, or try a different tokenizer"
                )
                prev_token_type = "SPECIAL"
                detokenized_explanation[len(detokenized)] = tokenized_explanation[counter]
                detokenized.append(
                    segmented_tokens[idx_special_tokens[last_special_token_pointer]]
                )
                last_special_token_pointer += 1
            counter += 1

        print("Detokenized (%03d): %s" % (len(detokenized), detokenized))
        print("Counter: %d" % (counter))

        if inputs_truncated:
            print("WARNING: Input truncated because of length, skipping check")
        else:
            assert counter == len(filtered_ids)
            assert len(detokenized) == len(original_tokens) + len(special_token_ids)
        print("===================================================================")

        return detokenized_explanation

    def _detokenize_explanation_old(self, sentence, tokenized_explanation, method="max"):
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

            # # Layer 0 is embedding
            # if layer == 0:
            #     self.interpreter = LayerIntegratedGradients(
            #         self.custom_forward, self.model.base_model.embeddings
            #     )
            # else:
            #     self.interpreter = LayerIntegratedGradients(
            #         self.custom_forward, self.model.base_model.encoder.layer[int(layer) - 1]
            #     )

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
                    sentence, tokenized_explanation, aggregation="max"
                ),
                "Average of subtokens": self._detokenize_explanation(
                    sentence, tokenized_explanation, aggregation="avg"
                ),
                "First Subtoken": self._detokenize_explanation(
                    sentence, tokenized_explanation, aggregation="first"
                ),
                "Last Subtoken": self._detokenize_explanation(
                    sentence, tokenized_explanation, aggregation="last"
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
    model = XLMRobertaForSequenceClassification.from_pretrained(args.model).to(device)
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
                f"Predicted class: {predicted_class} ({predicted_confidence * 100:.2f}%)"
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
