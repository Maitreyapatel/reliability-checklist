import logging
from typing import Any

import numpy as np
import torch
from transformers import pipeline

from .utils import get_model


class Model(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        model_type: str,
        additional_model_inputs: dict,
        decoder_model_name: str,
        model_path: str,
        tie_embeddings: bool,
        label: Any,
        tie_encoder_decoder: bool,
        tokenizer_data: dict,
        pipeline_name: str,
    ):
        super().__init__()
        self.is_generative_model = False if model_type == "discriminative" else True
        self.tokenizer_data = tokenizer_data
        self.additional_model_inputs = additional_model_inputs
        self.pipeline_name = pipeline_name

        self.model, self.tokenizer = get_model(
            model=model_type,
            model_name=model_name,
            tokenizer=None,
            decoder_model_name=decoder_model_name,
            model_path=model_path,
            tie_embeddings=tie_embeddings,
            label=label,
            tie_encoder_decoder=tie_encoder_decoder,
        )

        if self.pipeline_name:
            logging.warn("Currently pipeline only supports the CPU!!")
            self.nlp = pipeline(
                self.pipeline_name, model=self.model, tokenizer=self.tokenizer
            )
            self.candidates = list(self.tokenizer_data.label2id.keys())
            self.candidates2id = {v: en for en, v in enumerate(self.candidates)}
        else:
            if not self.is_generative_model:
                self.id2label = self.model.config.id2label
            else:
                self.label_inds = self.tokenizer.convert_tokens_to_ids(
                    list(self.tokenizer_data.label2id.values())
                )
                self.inds2label = {
                    k: v
                    for k, v in zip(
                        self.label_inds, list(self.tokenizer_data.label2id.keys())
                    )
                }

                self.inds2idx = {k: en for en, k in enumerate(sorted(self.label_inds))}
                self.idx2inds = {v: k for k, v in self.inds2idx.items()}

    def forward(self, inputs, labels):
        if self.pipeline_name:
            return self.nlp(
                self.tokenizer.batch_decode(inputs["input_ids"].cpu()),
                self.candidates,
                **self.additional_model_inputs
            )
        else:
            if self.additional_model_inputs:
                for k, v in self.additional_model_inputs.items():
                    inputs[k] = v

            if not self.is_generative_model:
                return self.model(**inputs)
            else:
                return self.model.generate(
                    **inputs
                )  # , **self.additional_model_inputs)

    def discriptive_postprocess(self, outputs, targets):
        preds = np.argmax(outputs.logits.cpu().numpy(), axis=1)
        p2u = [self.id2label[output.item()] for output in preds]
        return {
            "logits": outputs.logits,
            "p2u": {"predictions": p2u, "labels": targets},
        }

    def generative_postprocess(self, outputs, targets):
        scores = outputs.scores[0]
        logits = scores[:, self.label_inds]

        preds = np.argmax(logits.cpu().numpy(), axis=1)
        p2u = [self.inds2label[self.idx2inds[output.item()]] for output in preds]

        labels = torch.tensor([self.inds2idx[y] for y in targets.cpu().numpy()]).to(
            targets.device
        )
        return {
            "logits": logits,
            "p2u": {"predictions": p2u, "labels": labels},
        }

    def pipeline_postprocess(self, outputs, targets):
        logits = [out["scores"] for out in outputs]

        preds = np.argmax(logits, axis=1)
        p2u = [self.candidates[output] for output in preds]

        return {
            "logits": torch.tensor(logits),
            "p2u": {"predictions": p2u, "labels": targets},
        }

    def prediction2uniform(self, outputs, targets):
        if self.pipeline_name:
            results = self.pipeline_postprocess(outputs, targets)
        else:
            if self.is_generative_model:
                results = self.generative_postprocess(outputs, targets)
            else:
                results = self.discriptive_postprocess(outputs, targets)
        return results

    def input2uniform(self, batch):
        x, y = {}, {}
        for k, v in batch.items():
            if "label" == k:
                y[k] = v
            elif k in ["mapping", "primary_key", "augmentation"]:
                pass
            else:
                x[k] = v
        return x, y
