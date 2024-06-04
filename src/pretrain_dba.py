import os
import sys
import torch
import logging
import yaml
import json
import argparse
import warnings

from transformers import (
    AutoTokenizer,
    AutoAdapterModel,
    AutoModelForMaskedLM,
    AdapterConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    set_seed,
)
from transformers import AdapterTrainer as Trainer
from datasets import load_from_disk

from multiprocessing import cpu_count
from inclusivity_toolkit import eval_stereoset, eval_crows, eval_disco

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# set up logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("main")

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Found device: {device}")


def main(args):

    # Set seed
    logging.info(f"Setting seed to {args.training_seed}")
    set_seed(args.training_seed)

    # Load model and tokenizer
    logger.info(f"Loading {args.model_base_model} model from hub")
    model = AutoAdapterModel.from_pretrained(args.model_base_model)

    logging.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_tokenizer, use_fast=True)

    if not args.do_eval:
        # add adapter and head
        # set them as active and trainable
        if args.model_adapter_name is not None:
            config_dict = AdapterConfig.load(args.model_adapter_type).to_dict()
            config_dict["non_linearity"] = args.model_non_linearity
            config = AdapterConfig.load(config_dict)
            model.add_adapter(args.model_adapter_name, config=config)
            model.add_masked_lm_head(args.model_adapter_name)
            model.train_adapter([args.model_adapter_name])
            model.set_active_adapters(args.model_adapter_name)

        # Load dataset
        logger.info("Loading Dataset for fine-tuning")
        lm_dataset = load_from_disk(
            os.path.join(
                args.data_dir, f"{args.model_tokenizer}.{args.model_block_size}.hf"
            ),
            keep_in_memory=True,
        )

        # Initialize Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer,
            mlm=args.model_type == "mlm",
            mlm_probability=args.model_mlm_probability,
        )  # Data collators are used to create batches used for training (and evaluation)

        # Initialize Trainer
        logger.info("Initializing trainer")

        misc_training_args = {
            k.replace("training_", ""): v
            for k, v in vars(args).items()
            if k.startswith("training_")
        }

        training_args = TrainingArguments(
            output_dir=args.out_dir, fp16=(device == "cuda"), **misc_training_args
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=lm_dataset,
            data_collator=data_collator,
        )

        # Begin Training
        logger.info("Beginning Training")
        try:
            trainer.train()
        except KeyboardInterrupt:
            logger.info("Exiting Training")

        # Save trained model
        if args.model_adapter_name is not None:
            model.save_adapter(args.out_dir, args.model_adapter_name)
        else:
            model.save_pretrained(args.out_dir)

    else:
        # Load trained model
        if args.model_adapter_name is not None:
            logger.info(
                f"Loading debiasing adapter from {args.restore_from} and activating it"
            )
            model_adapter_name = model.load_adapter(args.restore_from, with_head=True)
            model.set_active_adapters(model_adapter_name)
        else:
            model = AutoModelForMaskedLM.from_pretrained(args.model_base_model)

        model.to(device)
        model.eval()
        logging.info(f"Model: {model}")

        # Evaluate Intrinsic Metrics
        with torch.no_grad():
            bias_eval_dict = {}
            if "stereoset" in args.bias_metrics:
                logger.info("Evaluating Stereoset")
                bias_eval_dict["stereoset"] = eval_stereoset(
                    model, tokenizer, model_type="encoder-only"
                )

            if "crows" in args.bias_metrics:
                logger.info("Evaluating CrowS")
                if args.model_type == "mlm":
                    bias_eval_dict["crows"] = eval_crows(model, tokenizer)
                else:
                    warnings.warn(
                        "CrowS does not support Causal LM models! Skipping this eval."
                    )

            logger.info(bias_eval_dict)
            logger.info(json.dumps(bias_eval_dict, indent=4))

        with open(os.path.join(args.restore_from, "bias_eval.json"), "w") as f:
            json.dump(bias_eval_dict, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, help="Path to the data directory")
    parser.add_argument("--out-dir", type=str, help="Path to the output directory")
    parser.add_argument(
        "--restore-from",
        type=str,
        help="Path to the directory containing the model to restore from",
    )
    parser.add_argument("--do-eval", action="store_true", help="to train or evaluate")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model-base-model", type=str, default="roberta-base")
    model_group.add_argument("--model-tokenizer", type=str, default="roberta-base")
    model_group.add_argument(
        "--model-type", type=str, default="mlm", choices=["mlm", "lm"]
    )
    model_group.add_argument("--model-mlm-probability", type=float, default=0.15)
    model_group.add_argument("--model-block-size", type=int, default=128)
    model_group.add_argument(
        "--model-adapter-type",
        type=str,
        default="pfeiffer",
        choices=["none", "pfeiffer", "lora", "prefix", "ia3"],
    )
    model_group.add_argument("--model-adapter-name", type=str, default=None)
    model_group.add_argument("--model-non-linearity", type=str, default="relu")

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--training-do-train", type=bool, default=True)
    training_group.add_argument(
        "--training-per-device-train-batch-size", type=int, default=512
    )
    training_group.add_argument(
        "--training-gradient-accumulation-steps", type=int, default=16
    )
    training_group.add_argument("--training-num-train-epochs", type=int, default=2)
    training_group.add_argument("--training-max-steps", type=int, default=-1)
    training_group.add_argument("--training-learning-rate", type=float, default=3e-5)
    training_group.add_argument("--training-weight-decay", type=float, default=0.01)
    training_group.add_argument(
        "--training-evaluation-strategy", type=str, default="no"
    )
    training_group.add_argument("--training-save-strategy", type=str, default="no")
    training_group.add_argument("--training-logging-steps", type=int, default=25000)
    training_group.add_argument("--training-seed", type=int, default=42)
    training_group.add_argument(
        "--training-dataloader-num-workers", type=int, default=cpu_count()
    )
    training_group.add_argument("--training-adam_beta1", type=float, default=0.9)
    training_group.add_argument("--training-adam_beta2", type=float, default=0.98)
    training_group.add_argument("--training-adam_epsilon", type=float, default=1e-6)
    training_group.add_argument("--training-warmup_ratio", type=int, default=0.1)
    training_group.add_argument("--training-report_to", type=str, default="none")

    bias_group = parser.add_argument_group("bias")
    bias_group.add_argument("--bias-metrics", type=str, default="stereoset,crows")

    args = parser.parse_args()
    logger.info(f"Running with: {args}")

    # Define the output directory to store model checkpoints
    if not args.do_eval:
        os.makedirs(args.out_dir, exist_ok=True)
        # Store the config file in the output directory
        with open(os.path.join(args.out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    # Run the main code
    main(args)
