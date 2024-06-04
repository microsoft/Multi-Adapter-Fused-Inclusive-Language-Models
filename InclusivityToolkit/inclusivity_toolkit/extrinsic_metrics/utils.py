import pandas as pd
from transformers import Pipeline
from torch.utils.data import Dataset
from scipy.special import softmax


class SentencePairDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self._data = self._get_dataset()

    def _get_dataset(self):
        df = pd.read_csv(self.data_path, sep="\t")
        return df.values.tolist()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


class SentencePairPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, text, second_text=None):
        return self.tokenizer.encode_plus(
            (text[0], text[1]),
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

    def _forward(self, model_inputs):
        return self.model(**model_inputs)

    def postprocess(self, model_outputs):
        logits = model_outputs.logits[0].numpy()
        outputs = softmax(logits) if logits.shape[0] > 1 else logits[0]
        return outputs.tolist()
