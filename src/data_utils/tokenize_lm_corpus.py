import os
import argparse
from transformers import AutoTokenizer
from datasets import load_dataset
from datasets import Dataset
import dill as pickle
from multiprocessing import cpu_count

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def tokenize_and_group_dataset(tokenizer, dataset=None, block_size=128):
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def tokenize(examples):
        return tokenizer([x for x in examples["text"]], truncation=True)

    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=1024,
        num_proc=cpu_count(),
        remove_columns=dataset.column_names,
    )

    lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=cpu_count())

    return lm_dataset


# python -m src.data_utils.tokenize_lm_corpus --txt_file {path_to_store_cda_data}/{output_filename}
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--txt_file",
        type=str,
        required=True,
        help="The output for the counterfactual augmented dataset.",
    )
    parser.add_argument(
        "--tokenizer_variant",
        type=str,
        default="bert-base-uncased",
    )
    parser.add_argument("--block_size", type=int, default=128)

    args = parser.parse_args()

    data_dir = os.path.dirname(args.txt_file)
    data_file = args.txt_file.split("/")[-1]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_variant, use_fast=True)
    text_data = []

    print("####", data_dir)

    dataset = load_dataset("text", data_files=args.txt_file)["train"]
    lm_dataset = tokenize_and_group_dataset(
        tokenizer, dataset, block_size=args.block_size
    )
    if not os.path.exists(f"{data_dir}/tokenized/"):
        os.makedirs(f"{data_dir}/tokenized/")
    lm_dataset.save_to_disk(
        f"{data_dir}/tokenized/{args.tokenizer_variant}.{args.block_size}.hf"
    )


if __name__ == "__main__":
    main()
