import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from itertools import chain, combinations
from inclusivity_toolkit import eval_stsb_mnli, eval_stsb
from transformers import (
    AutoTokenizer,
    AutoAdapterModel,
)
from transformers.adapters.composition import Fuse, Stack


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
    task = "stsb"
    languages = ["English"]
    root_dir = f"/home/t-assathe/local_ckpts_stacked_bert/fusion_{task}/"
    bias_stsb_root_dir = (
        "/home/t-assathe/InclusivityToolkit/inclusivity_toolkit/evaluators/EXTRINSIC/"
    )
    stsb_root_dir = "/home/t-assathe/stsb_translated/"
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
    all_dims = ["gender", "race", "religion"]
    for lang in languages:
        results = pd.DataFrame(
            columns=["model_name", "accuracy"] + all_dims + ["average", "acc*(1-avg)"]
        )
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
            dim_to_df = {}
            new_dict = {"model_name": bias_dir}
            if lang == "English":
                with open(f"{bias_root_dir}/eval_results.json") as f:
                    new_dict["accuracy"] = json.load(f)["eval_accuracy"]
            else:
                new_dict["accuracy"] = eval_stsb(
                    model=model,
                    tokenizer=tokenizer,
                    test_set_path=f"{stsb_root_dir}/val.{lang}",
                    block_size=128,
                    batch_size=4096,
                )
            for dim in all_dims:
                df = eval_stsb_mnli(
                    model=model,
                    tokenizer=tokenizer,
                    dirpath=f"{bias_stsb_root_dir}/data_{lang}/",
                    dimension=dim,
                    batch_size=4096,
                )
                comb_columns = list(combinations(list(df.columns), 2))
                average = 0
                for column_x, column_y in comb_columns:
                    diff = abs(df[column_x] - df[column_y])
                    df[f"{column_x}::{column_y}"] = diff
                    average += diff
                df["overall_delta"] = average / len(comb_columns)
                dim_to_df[dim] = df
            with pd.ExcelWriter(
                f"{output_root_dir}/all_dims_processed_{lang}.xlsx"
            ) as writer:
                avg = 0
                for dim, df in dim_to_df.items():
                    res = df["overall_delta"].mean()
                    df.to_excel(writer, sheet_name=dim)
                    new_dict[dim] = res
                    avg += res
                avg /= len(dim_to_df.keys())
                new_dict["average"] = avg
            new_dict["acc*(1-avg)"] = new_dict["accuracy"] * (1 - new_dict["average"])
            print(new_dict)
            results = pd.concat([results, pd.DataFrame([new_dict])], ignore_index=True)
        results.set_index("model_name")
        results.to_csv(f"{base_model}_{lang}.csv")


if __name__ == "__main__":
    main()
