import numpy as np
import torch
from transformers import AutoModelForSequenceClassification


class Model(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def forward(self, inputs):
        return self.model(**inputs)

    def prediction2uniform(self, outputs):
        preds = np.argmax(outputs.logits.cpu().numpy(), axis=1)
        p2u = [self.model.config.id2label[output.item()] for output in preds]
        return {"raw": outputs, "p2u": p2u}

    def input2uniform(self, batch):
        x, y = {}, {}
        for k, v in batch.items():
            if "label" == k:
                y[k] = v
            else:
                x[k] = v
        return x, y