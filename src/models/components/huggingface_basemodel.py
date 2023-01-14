import numpy as np
import torch
from .utils import get_model
from typing import Any


class Model(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        model_type: str,
        decoder_model_name: str,
        model_path: str,
        tie_embeddings: bool,
        label: Any,
        tie_encoder_decoder: bool,
    ):
        super().__init__()
        self.is_generative_model = False if model_type == "discriminative" else True

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

    def forward(self, inputs, labels):
        return self.model(**inputs, labels=labels)

    def discriptive_postprocess(self, outputs, targets):
        preds = np.argmax(outputs.logits.cpu().numpy(), axis=1)
        p2u = [self.model.config.id2label[output.item()] for output in preds]
        return {
            "logits": outputs.logits,
            "p2u": {"predictions": p2u, "labels": targets},
        }

    def generative_postprocess(self, outputs, targets):
        predictions = self.tokenizer.batch_decode(
            outputs.preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        labels = self.tokenizer.batch_decode(
            targets, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return {
            "logits": outputs.logits,
            "p2u": {"predictions": predictions, "labels": labels},
        }

    def prediction2uniform(self, outputs, targets):
        if self.is_generative_model:
            results = self.generative_postprocess(outputs, targets)
        else:
            results = self.discriptive_postprocess(outputs, targets)
        return results

    def input2uniform(self, batch):
        ## TODO: this will change for the generative models
        x, y = {}, {}
        for k, v in batch.items():
            if "label" == k:
                y[k] = v
            elif "augmentation" == k:
                pass
            else:
                x[k] = v
        return x, y
