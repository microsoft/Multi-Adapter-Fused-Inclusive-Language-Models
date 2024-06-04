from transformers import (
    AutoTokenizer,
    AutoAdapterModel,
    TrainingArguments,
    AdapterConfig,
    AdapterTrainer,
    Trainer,
)
from transformers.adapters.composition import Fuse, Stack
from typing import List
from dataclasses import dataclass, asdict, field
from simple_parsing import ArgumentParser
from datetime import datetime
from loguru import logger as lgr
import torch
import json
import sys
import os
from datetime import datetime
from inclusivity_toolkit import eval_stsb_mnli
from utils import (
    load_datasets,
    compute_metrics_mnli,
    compute_metrics_stsb,
    compute_metrics_jigsaw,
)

logger_config = {
    "handlers": [
        {"sink": sys.stdout, "format": "{time} --- {message}"},
        {"sink": "loguru.train.serialized.log", "serialize": True},
        {"sink": "loguru.train.log", "format": "{time} --- {message}"},
    ]
}


def dataclass_factory(base, name, exclusions, exclude_privates=True):
    from dataclasses import make_dataclass, fields

    new_fields = [(i.name, i.type, i) for i in fields(base) if i.name not in exclusions]
    new_fields = (
        list(filter(lambda x: not x[0].startswith("_"), new_fields))
        if exclude_privates
        else new_fields
    )
    return make_dataclass(name, new_fields)


@dataclass
class Options:
    save_dir_fusion: str = "./ckpts_fusion/"
    timestamp_save_dir: bool = False
    debiasing_adapter_paths: List[str] = field(default_factory=[].copy)
    adapter_fusion_path: str = ""
    add_task_adapter: bool = True
    task_adapter_type: str = "pfeiffer"
    task_adapter_activation: str = "silu"
    model_name: str = "bert-base-multilingual-uncased"
    tokenizer_name: str = "bert-base-multilingual-uncased"
    expt_name: str = "expt"
    task_name: str = "stsb"
    block_size: int = 128


def get_output_dir(general_args):
    if general_args.expt_name == "fusion:":
        general_args.save_dir_fusion += "/" + general_args.model_name.split("/")[-1]
    if general_args.timestamp_save_dir:
        out_dir = general_args.save_dir_fusion + datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
    else:
        out_dir = general_args.save_dir_fusion
    return out_dir


@lgr.catch()
def main():
    parser = ArgumentParser()
    parser.add_arguments(Options, dest="general")
    parser.add_arguments(
        dataclass_factory(
            TrainingArguments, "TrainerOptions", exclusions=["output_dir"]
        ),
        dest="trainer",
    )
    args = parser.parse_args()

    general_args = args.general
    trainer_args = args.trainer

    output_dir = get_output_dir(general_args)
    os.makedirs(output_dir, exist_ok=True)
    logger_config["handlers"].append(
        {
            "sink": output_dir
            + "/"
            + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            + ".log",
            "format": "{time} --- {message}",
        },
    )
    lgr.configure(**logger_config)

    lgr.info("General Options")
    lgr.info("\n".join(f'"{k}":"{v}"' for k, v in asdict(general_args).items()))
    lgr.info("Training Options")
    lgr.info("\n".join(f'"{k}":"{v}"' for k, v in asdict(trainer_args).items()))
    training_args = TrainingArguments(output_dir=output_dir, **asdict(trainer_args))

    model = AutoAdapterModel.from_pretrained(general_args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(general_args.tokenizer_name)

    # Load debiasing adapters and potentially the fusion as well
    # If no biases are provided, then experiment would just be `fusion:`
    # otherwise it would be `fusion:race+religion+profession` etc.
    names = []
    if general_args.expt_name != "fusion:":
        for dba_dir in general_args.debiasing_adapter_paths:
            lgr.info(f'Loading a single debiasing adapter from "{dba_dir}"')
            debiasing_adapter_name = model.load_adapter(dba_dir, with_head=False)
            lgr.info(f"Loaded adapter named {debiasing_adapter_name}")
            names.append(debiasing_adapter_name)
            lgr.info(
                f"debiasing_adapter_name: {debiasing_adapter_name}, names = {names}"
            )

        if len(names) > 1:
            # We have a fusion
            if len(general_args.adapter_fusion_path) > 0:
                lgr.info(
                    f"Loading pretrained fusion from {general_args.adapter_fusion_path}"
                )
                fused_adapters = model.load_adapter_fusion(
                    general_args.adapter_fusion_path, "adapter_fusion_layer"
                )
                lgr.info(f"fused_adapters: {fused_adapters}")
            else:
                lgr.info(f"Fusing {names}")
                fused_adapters = Fuse(*names)
                model.add_adapter_fusion(fused_adapters)
                lgr.info(f"fused_adapters:{fused_adapters}")

    if general_args.add_task_adapter:
        config_dict = AdapterConfig.load(general_args.task_adapter_type).to_dict()
        config_dict["non_linearity"] = general_args.task_adapter_activation
        config = AdapterConfig.load(config_dict)
        model.add_adapter(general_args.task_name, config=config)
        lgr.info(f"Added a task adapter named {general_args.task_name}")

    num_labels = {"stsb": 1, "mnli": 3, "jigsaw": 1}[general_args.task_name]
    compute_metrics_fn = {
        "stsb": compute_metrics_stsb,
        "mnli": compute_metrics_mnli,
        "jigsaw": compute_metrics_jigsaw,
    }[general_args.task_name]

    # add task specific classification head
    model.add_classification_head(general_args.task_name, num_labels=num_labels)
    if len(names) > 1:
        # We have fusion of multiple debiasing adapters
        model.train_adapter_fusion(fused_adapters)
        if general_args.add_task_adapter:
            model.train_adapter(general_args.task_name)
            model.set_active_adapters(Stack(fused_adapters, general_args.task_name))
        else:
            model.set_active_adapters(fused_adapters)
    elif len(names) == 1:
        # We only have the single debiasing adapter
        model.train_adapter(debiasing_adapter_name)
        if general_args.add_task_adapter:
            model.train_adapter(general_args.task_name)
            model.set_active_adapters(
                Stack(debiasing_adapter_name, general_args.task_name)
            )
        else:
            model.set_active_adapters(debiasing_adapter_name)
    else:
        if general_args.add_task_adapter:
            model.train_adapter(general_args.task_name)
            model.set_active_adapters(general_args.task_name)

    # Activate the head
    model.active_head = general_args.task_name
    lgr.info(f"Model::Active Adapter -- {str(model.active_adapters)}")
    lgr.info(f"Model::Active Head -- {str(model.active_head)}")

    # Disable EVERYTHING else except the fusion and the head
    # Despite calling `train_adapter`, we don't actually want to train
    # the loaded adapter. `train_adapter` is required for setting proper
    # conditions for the overall model
    for name, module in model.named_modules():
        for _name, _param in module.named_parameters():
            _param.requires_grad = (
                general_args.task_name in name or "adapter_fusion_layer" in name
            )

    for name, module in model.named_modules():
        for _name, _param in module.named_parameters():
            if _param.requires_grad:
                lgr.info(f'Gradient enabled for _name = "{_name}", name = "{name}"')

    lgr.info(f"Model: {model}")

    # load the required task dataset
    lgr.info("Loading datasets")
    train_dataset, eval_dataset = load_datasets(
        tokenizer=tokenizer,
        task_name=general_args.task_name,
        block_size=general_args.block_size,
    )

    # Using AdapterTrainer is better
    TrainerClass = (
        AdapterTrainer if general_args.add_task_adapter or len(names) > 0 else Trainer
    )
    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_fn,
    )

    try:
        result = trainer.evaluate()
        lgr.info("Before training")
        lgr.info(result)
        lgr.info("Beginning training")
        result = trainer.train()
    except KeyboardInterrupt:
        lgr.info("Exited training due to keyboard interrupt")

    # Save the head manually
    # This is useful when we're using HF's Trainer and not AdapterTrainer
    # AdapterTrainer can overwrite this checkpoint and it's totally fine
    model.save_head(
        os.path.join(output_dir, general_args.task_name), general_args.task_name
    )
    trainer.save_model()
    trainer.save_state()
    lgr.info("Saved the model and the trainer state")

    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    lgr.info(result.metrics)
    lgr.info("Saved the train metrics")

    result = trainer.evaluate()
    trainer.log_metrics("eval", result)
    trainer.save_metrics("eval", result)
    lgr.info(result)
    lgr.info("Saved the eval metrics")

    trainer.model.eval()
    if general_args.task_name in ["stsb", "mnli"]:
        with torch.no_grad():
            for dim in ["gender", "race", "religion"]:
                df = eval_stsb_mnli(
                    model=trainer.model,
                    tokenizer=tokenizer,
                    dirpath="InclusivityToolkit/inclusivity_toolkit/evaluators/EXTRINSIC/data_English",
                    dimension=dim,
                    batch_size=256,
                )
                results = df.mean().to_dict()
                results["overall"] = sum(results.values()) / len(results)
                results["active_adapters"] = str(model.active_adapters)
                results["active_head"] = str(model.active_head)
                lgr.info(results)
                df.to_csv(f"{output_dir}/raw_dim={dim}.csv")
                with open(f"{output_dir}/dim={dim}.json", "w") as f:
                    json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
