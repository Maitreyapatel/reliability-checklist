from transformers import AutoTokenizer, T5Tokenizer


class conversion_process:
    def __init__(self, conversion_map) -> None:
        super().__init__()
        self.conversion_map = conversion_map

    def process(self, examples):
        label = []
        for exp in examples["label"]:
            label.append(self.conversion_map[exp])
        return {"label": label}


class general_tokenization:
    def __init__(
        self,
        model_name: str,
        model_type: str,
        is_generative_model: bool,
        tokenizer_args: dict,
        data_processing: dict,
        label2id: dict,
        cols: list,
        label_col: str,
    ):
        if model_type != "t5":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.is_generative_model = is_generative_model
        self.data_processing = data_processing
        self.tokenizer_args = tokenizer_args
        self.label2id = label2id
        self.label_col = label_col
        self.cols = cols

    def process(self, example):
        return self.tokenizer(example["input_data"], **self.tokenizer_args)


def process_label2id(gt_label2id, pred_label2id):
    assert len(gt_label2id) == len(pred_label2id)

    dataset_converion = {}
    for i in list(gt_label2id.keys()):
        dataset_converion[i] = pred_label2id[gt_label2id[i]]
    return dataset_converion
