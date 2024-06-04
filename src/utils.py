import jigsaw
import pandas as pd
from datasets import load_dataset
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
from transformers import EvalPrediction


# function to compute metrics for sts-b task
def compute_metrics_stsb(pred: EvalPrediction) -> dict:
    preds = [p[0] for p in pred.predictions]
    corr, _ = pearsonr(preds, pred.label_ids)
    return {"accuracy": corr}


# function to compute metrics for mnli task
def compute_metrics_mnli(pred: EvalPrediction) -> dict:
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


# function to compute metrics for jigsaw task
def compute_metrics_jigsaw(pred: EvalPrediction) -> dict:
    # This is a hardcoded path for sanity sake. If you're doing anything custom, please do NOT use this function
    df = pd.read_csv("/home/t-assathe/jigsaw/test_public_leaderboard_from_hf.csv")
    MODEL_NAME = "predictions"
    df = jigsaw.convert_dataframe_to_bool(df)
    df[MODEL_NAME] = [p[0] for p in pred.predictions]
    bias_metrics_df = jigsaw.compute_bias_metrics_for_model(
        df, jigsaw.identity_columns, MODEL_NAME, jigsaw.TOXICITY_COLUMN
    )
    return {
        "accuracy": jigsaw.get_final_metric(
            bias_metrics_df, jigsaw.calculate_overall_auc(df, MODEL_NAME)
        )
    }


# generic tokenization function
def tokenize_function_stsb(example, tokenizer, max_length=128):
    return tokenizer(
        example["sentence1"],
        example["sentence2"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )


# generic tokenization function
def tokenize_function_mnli(example, tokenizer, max_length=128):
    return tokenizer(
        example["premise"],
        example["hypothesis"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )


def tokenize_function_jigsaw(example, tokenizer, max_length=128):
    return tokenizer(
        example["comment_text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )


# generic function to load stsb/mnli datasets
def load_datasets(tokenizer, task_name="stsb", block_size=128):
    if task_name == "jigsaw":
        dataset = load_dataset(
            "jigsaw_unintended_bias", data_dir="/home/t-assathe/jigsaw/"
        )

        def generate_jigsaw_label(x):
            return dict(**x, label=1.0 if x["target"] >= 0.5 else 0.0)

        dataset = dataset.map(generate_jigsaw_label)
    else:
        dataset = load_dataset("glue", task_name)

    f = {
        "stsb": tokenize_function_stsb,
        "mnli": tokenize_function_mnli,
        "jigsaw": tokenize_function_jigsaw,
    }[task_name]

    valid_name = {
        "stsb": "validation",
        "mnli": "validation_matched",
        "jigsaw": "test_public_leaderboard",
    }[task_name]

    tokenized_dataset = dataset.map(
        lambda examples: f(examples, tokenizer=tokenizer, max_length=block_size),
        batched=True,
    )

    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset[valid_name]
    return train_dataset, eval_dataset
