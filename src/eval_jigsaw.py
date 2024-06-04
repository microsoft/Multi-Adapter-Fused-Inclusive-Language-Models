import os
import json
import torch
import jigsaw
import pandas as pd
from tqdm import tqdm
from itertools import chain, combinations
from functools import partial
from inclusivity_toolkit import eval_stsb_mnli, eval_stsb
from utils import load_datasets
from transformers import (
    AutoTokenizer,
    AutoAdapterModel,
    Trainer,
    TrainingArguments,
)
from transformers.adapters.composition import Fuse, Stack


class Jigsaw(torch.utils.data.Dataset):
    def __init__(self, path, **kwargs):
        super().__init__(**kwargs)
        self.data = pd.read_csv(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]


def collate_fn_jigsaw(batch, max_length, tokenizer):
    texts = [item["comment_text"] for item in batch]
    ret = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    ret["label"] = torch.Tensor(
        [1.0 if item["target"] >= 0.5 else 0.0 for item in batch]
    )
    return ret


def model_to_bias_name(base_model, bias):
    return {
        "bert-base-uncased": "cda_debiasing_adapter"
        if bias == "gender"
        else f"cda_{bias}_debiaising_adapter",
        "bert-base-multilingual-uncased": f"debiasing_{bias}",
        "roberta-base": f"cda_{bias}_debiasing_adapter",
        "xlm-roberta-base": f"xlmr_cda_debiasing_adapter_{bias}",
    }[base_model]


def powerset(s):
    # Potential perpyu addition
    s = list(s)
    return chain.from_iterable(combinations(s, l) for l in range(len(s) + 1))


def prepare_model(base_model, biases, root_dir, task, task_adapter=False):
    model = AutoAdapterModel.from_pretrained(base_model)
    for bias in biases:
        name = model.load_adapter(root_dir + "/" + bias)
        print(name)
    if len(biases) > 1:
        name = Fuse(*biases)
        model.add_adapter_fusion(name)
        model.load_adapter_fusion(root_dir + "/" + ",".join(biases))
    if task_adapter:
        task_name = model.load_adapter(root_dir + "/" + task, with_head=False)
        if len(biases) > 0:
            # some DBA or Fusion of DBA exists already
            model.set_active_adapters(Stack(name, task_name))
        else:
            # No DBA exists so just activate the task adapter
            model.set_active_adapters(task_name)
    else:
        # If there's no task adapter, name will contain either the fusion or the single DBA
        model.set_active_adapters(name)
    name = model.load_head(root_dir + "/" + task)
    model.active_head = task
    print(model.active_adapters)
    print(model.active_head)
    model.eval()
    return model


def main():
    base_model = "bert-base-uncased"
    task = "jigsaw"
    root_dir = f"/home/t-assathe/local_ckpts_stacked_bert/fusion_{task}/"
    jigsaw_dir = "/home/t-assathe/jigsaw/"
    results_dir = f"/home/t-assathe/BlobStorage/modular-rai-gcr-blob-abs/extrinsic_results/stacked_bert/{task}/"
    task_adapter = True
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    # Add Pad token if it doesn't exist
    if tokenizer.pad_token_id is None:
        if model.config.eos_token_id is not None:
            tokenizer.pad_token_id = model.config.eos_token_id
        else:
            tokenizer.add_tokens("[PAD]", special_tokens=True)
            tokenizer.pad_token_id = tokenizer.get_added_vocab()["[PAD]"]
            model.resize_token_embeddings(len(tokenizer))
    all_biases = ["gender", "race", "religion", "profession"]
    columns = ["subgroup_size", "subgroup_auc", "bpsn_auc", "bnsp_auc"]
    results = pd.DataFrame()
    test_jigsaw = Jigsaw(f"{jigsaw_dir}/test_from_hf.csv")
    for biases in tqdm(powerset(all_biases)):
        bias_dir = "+".join(biases) if len(biases) > 0 else base_model
        bias_root_dir = f"{root_dir}/{bias_dir}/"
        print(biases, bias_root_dir)
        model = prepare_model(
            base_model,
            [model_to_bias_name(base_model, bias) for bias in biases],
            bias_root_dir,
            task,
            task_adapter,
        )
        output_root_dir = f"{results_dir}/{bias_dir}/"
        os.makedirs(output_root_dir, exist_ok=True)
        trainer = Trainer(
            model,
            data_collator=partial(
                collate_fn_jigsaw, tokenizer=tokenizer, max_length=128
            ),
            args=TrainingArguments(
                report_to=None,
                output_dir="./tmp_results/",
                overwrite_output_dir=True,
                per_device_eval_batch_size=4096,
            ),
        )
        preds = trainer.predict(test_jigsaw)
        df = pd.read_csv(f"{jigsaw_dir}/test_from_hf.csv")
        MODEL_NAME = "predictions"
        df = jigsaw.convert_dataframe_to_bool(df)
        df[MODEL_NAME] = [p[0] for p in preds.predictions]
        bias_metrics_df = jigsaw.compute_bias_metrics_for_model(
            df, jigsaw.identity_columns, MODEL_NAME, jigsaw.TOXICITY_COLUMN
        )
        bias_metrics_df.to_csv(f"{results_dir}/{bias_dir}/bias_metrics_df.csv")
        model_results = {
            "model": bias_dir,
            "active_adapters": str(model.active_adapters),
            "active_head": str(model.active_head),
        }
        for index, row in bias_metrics_df.iterrows():
            for column in columns:
                model_results[f'{row["subgroup"]}_{column}'] = row[column]
        model_results["overall_score"] = jigsaw.get_final_metric(
            bias_metrics_df, jigsaw.calculate_overall_auc(df, MODEL_NAME)
        )
        results = pd.concat([results, pd.DataFrame([model_results])], ignore_index=True)
    results = results.set_index("model")
    results.to_csv(f"{task}.csv")


if __name__ == "__main__":
    main()
