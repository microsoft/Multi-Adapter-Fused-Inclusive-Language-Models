import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import SentencePairDataset, SentencePairPipeline
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import pipeline, AutoModelForSequenceClassification

PIPELINE_REGISTRY.register_pipeline(
    "sentence-pair-classification",
    pipeline_class=SentencePairPipeline,
    pt_model=AutoModelForSequenceClassification,
)

torch.set_float32_matmul_precision("high")
device = "cuda" if torch.cuda.is_available() else "cpu"

TOOLKIT_DIR = os.path.split(os.path.dirname(__file__))[0]
DATA_DIR = os.path.join(TOOLKIT_DIR, "evaluators/EXTRINSIC/data/")


def eval_stsb_mnli(
    model,
    tokenizer,
    dirpath=None,
    block_size=128,
    batch_size=4096,
    dimension="gender",
    task="stsb",
):

    pipe = pipeline(
        task="sentence-pair-classification",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        get_framework="pt",
        truncation=True,
        max_length=block_size,
        padding="max_length",
        device=0,
    )

    test_set_path = os.path.join(DATA_DIR if dirpath is None else dirpath, dimension)
    files = os.listdir(test_set_path)
    N, results = len(files), {}

    # stsb/nli needs N files in a directory
    # and code will compute the similarity between the two
    # sentences in that file and save the results in a csv file
    # you can use the csv file to compute pairwise absolute difference between the similarity scores
    for i, file_ in enumerate(files):
        print(f"[{i+1}/{N}] Evaluating {file_} ...")
        name = file_.split(".")[0]
        dataset = SentencePairDataset(os.path.join(test_set_path, file_))
        output = [out for out in tqdm(pipe(dataset, batch_size=batch_size))]

        if task == "mnli":
            output = [",".join(out) for out in output]

        results[name] = output

    return pd.DataFrame(results)
