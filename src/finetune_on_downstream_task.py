import argparse
import logging
import os
import torch
import sys
import uuid
import yaml

from multiprocessing import cpu_count
from transformers import (
    AutoTokenizer,
    AutoAdapterModel,
    set_seed,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
)

from transformers.adapters.composition import Fuse
from inclusivity_toolkit import SentencePairPipeline, eval_stsb_mnli

from utils import (
    load_datasets,
    compute_metrics_mnli,
    compute_metrics_stsb,
)

from transformers.pipelines import PIPELINE_REGISTRY

PIPELINE_REGISTRY.register_pipeline(
    "sentence-pair-classification",
    pipeline_class=SentencePairPipeline,
    pt_model=AutoModelForSequenceClassification,
)

torch.set_float32_matmul_precision("high")
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


# training function
def main(args):

    logging.info(f"Setting seed to {args.seed}")
    set_seed(args.seed)

    # load tokenizer, preferrably fast version
    logger.info("Initializing tokenizer (fast version)")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

    # load the model
    if args.restore_from is not None and os.path.exists(
        os.path.join(args.restore_from, "model_checkpoint")
    ):
        # if a checkpoint is provided, load it
        logger.info(f"Loading {args.model_name} from {args.restore_from}")
        model = AutoAdapterModel.from_pretrained(
            os.path.join(args.restore_from, "model_checkpoint")
        )
    else:
        # or else, load it from hub
        logger.info(f"Loading {args.model_name} model from hub")
        model = AutoAdapterModel.from_pretrained(args.model_name)

    # this is for fusion. We provide the in_dir as path separated by :
    # each path corresponds to the checkpoint of a pre-trained debiasing adapter
    paths = args.dba_dir.split(":") if args.dba_dir is not None else []
    # or else you can even pass the flag --use_fusion
    # to use the fusion layer during evaluation.
    # the code will automatically look for the debiasing adapters
    # in the restore_from directory

    if not args.do_eval:
        # training
        # load model from scratch
        # load debaising adapter from path
        if args.add_debiasing_adapter:
            if len(paths) > 1:
                # if fusion is available, load all the adapters and fuse them
                # collect the names of the debiasing adapters as you fuse them
                # as we need them for identifying the fusion layer later
                logger.info(f"Loading debiasing adapters from {paths}")
                names = [model.load_adapter(path, with_head=False) for path in paths]
                if args.fusion_dir is None:
                    logger.info(f"Fusing {names}")
                    fused_adapters = Fuse(*names)
                    model.add_adapter_fusion(fused_adapters)
                else:
                    logger.info(f"Loading fusion layer from {args.fusion_dir}")
                    fused_adapters = model.load_adapter_fusion(args.fusion_dir)
            else:
                # if no fusion is available, load the debiasing adapter
                logger.info(f"Loading debiasing adapter from {args.dba_dir}")
                debiasing_adapter_name = model.load_adapter(
                    args.dba_dir, with_head=False
                )

        # add the task-specific adapter to the model
        if args.add_task_adapter:
            logger.info(f"Adding {args.task_adapter_type} task adapters")
            model.add_adapter(args.task_name, config=args.task_adapter_type)

        num_labels = {"stsb": 1, "mnli": 3}[args.task_name]

        compute_metrics_fn = {
            "stsb": compute_metrics_stsb,
            "mnli": compute_metrics_mnli,
        }[args.task_name]

        # add task specific head to the model
        logger.info(
            f"Adding {args.task_name} classification head with num_labels={num_labels}"
        )
        model.add_classification_head(args.task_name, num_labels=num_labels)

        # set the required adapters to train and as active
        # the order should be:
        # 1. set adapters to train
        # 2. set adapters as active
        if args.add_debiasing_adapter and args.add_task_adapter:
            logger.info("Setting only task adapter to train")
            if len(paths) > 1:
                if args.fusion_dir is None:
                    logger.info(
                        "Fused debiasing adapters and task adapters found. Stacking both adapters and activating them"
                    )
                    # a quick hack to only train the fusion layer and task adapter
                    # the debiasing adapter are not trained
                    for name, module in model.named_modules():
                        for name_, param_ in module.named_parameters():
                            param_.requires_grad = (
                                args.task_name in name or "adapter_fusion_layer" in name
                            )
                    model.set_active_adapters([fused_adapters, args.task_name])
                else:
                    logger.info(
                        "Pretrained Fusion layer and task adapters found. Training only task adapters and setting them as active"
                    )
                    model.train_adapter([args.task_name])
                    model.set_active_adapters(args.task_name)
            else:
                logger.info(
                    "Debiasing and task adapters found. Stacking both adapters and activating them"
                )
                model.train_adapter([args.task_name])
                model.set_active_adapters([debiasing_adapter_name, args.task_name])
        elif args.add_debiasing_adapter:
            # Similar to ADELE paper
            # we train the entire model + debiasing adapter (or fusion)
            # for extrinsic evaluation.
            # Note: if we have only one adapter
            # we can directly train the adapter
            # it will automatically be activated
            if len(paths) > 1:
                if args.fusion_dir is None:
                    logger.info(
                        "Setting model, fusion layer, and debiasing adapters to train"
                    )
                    model.train_adapter_fusion(fused_adapters, unfreeze_adapters=True)
            else:
                logger.info("Setting model and debiasing adapter to train")
                model.train_adapter([debiasing_adapter_name])
            model.freeze_model(False)
        elif args.add_task_adapter:
            logger.info("Setting only task adapter to train")
            model.train_adapter([args.task_name])

        # load the required task dataset
        logger.info("Loading datasets")
        train_dataset, eval_dataset = load_datasets(
            tokenizer=tokenizer, task_name=args.task_name, block_size=args.block_size
        )

        training_args = TrainingArguments(
            output_dir=args.save_to,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            logging_steps=args.logging_steps,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            run_name=args.run_id,
            label_names=["labels"],
            remove_unused_columns=False,
            save_total_limit=args.save_total_limit,
            dataloader_num_workers=cpu_count(),
            greater_is_better=True,
            metric_for_best_model="eval_accuracy",
            learning_rate=args.learning_rate,
            report_to="none",
            load_best_model_at_end=True,
            overwrite_output_dir=True,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            fp16=(device == "cuda"),
        )

        # before training, make sure the head is also active.
        model.active_head = args.task_name

        # better to log the model and check if everything is fine
        logger.info(f"model: {model}")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics_fn,
        )

        # Begin Training
        logger.info("Beginning Training")
        try:
            trainer.train()
        except KeyboardInterrupt:
            logger.info("Exiting Training")

        # save the model, adapters and head
        logger.info("Training completed")
        if args.add_task_adapter:
            logger.info(
                f"Saving only task adapter and classification head to {args.save_to}"
            )
            save_to_task_adapter = os.path.join(args.save_to, "task_adapter_checkpoint")
            model.save_adapter(save_to_task_adapter, args.task_name, with_head=False)
        elif args.add_debiasing_adapter:
            if len(paths) > 1:
                # if fusion is available, we save the fusion, and each of the participating adapters
                # the naming is adapter#0, adapter#1, ..., adapter#n
                # but when loading, the model will automatically load them with correct names
                logger.info(
                    f"No task adapter found. Saving fused debiasing adapter, classification head and model to {args.save_to}"
                )
                save_to_debiasing_adapter = os.path.join(
                    args.save_to, "fusion_adapter_checkpoint"
                )
                model.save_adapter_fusion(
                    save_to_debiasing_adapter, names, with_head=False
                )
                for i, name in enumerate(names):
                    model.save_adapter(
                        os.path.join(args.save_to, f"adapter#{i}"),
                        name,
                        with_head=False,
                    )
            else:
                logger.info(
                    f"No task adapter found. Saving model, classification_head and debiasing adapter to {args.save_to}"
                )
                save_to_debiasing_adapter = os.path.join(
                    args.save_to, "debiasing_adapter_checkpoint"
                )
                model.save_adapter(
                    save_to_debiasing_adapter, debiasing_adapter_name, with_head=False
                )
            save_to_model = os.path.join(args.save_to, "model_checkpoint")
            model.save_pretrained(save_to_model)
        else:
            logger.info(
                f"No task or debiasing adapter found. Saving model and classification_head {args.save_to}"
            )
            save_to_model = os.path.join(args.save_to, "model_checkpoint")
            model.save_pretrained(save_to_model)

        # save the head in all cases at the end
        save_to_head = os.path.join(args.save_to, "head_checkpoint")
        model.save_head(save_to_head, args.task_name)

    # evaluation
    else:
        # since we already loaded the model and fusion participants if available
        # load the rest of the adapters and set them as active
        logger.info("evaluating model")
        if args.add_debiasing_adapter and args.add_task_adapter:
            logger.info(f"Loading task adapter from {args.restore_from}")
            task_adapter_name = model.load_adapter(
                os.path.join(args.restore_from, "task_adapter_checkpoint")
            )
            if len(paths) > 1:
                logger.info(f"Loading fusion adapters from {args.dba_dir}")
                names = [model.load_adapter(p, with_head=False) for p in paths]
                fusion_adapter_name = Fuse(*names)
                if args.fusion_dir is None:
                    model.add_adapter_fusion(fusion_adapter_name)
                else:
                    logger.info(f"Loading fusion from {args.fusion_dir}")
                    model.load_adapter_fusion(args.fusion_dir, with_head=False)
            else:
                debiasing_adapter_name = model.load_adapter(
                    args.dba_dir, with_head=False
                )
        elif args.add_debiasing_adapter:
            logger.info(f"Loading debiasing adapter from {args.restore_from}")
            if len(paths) > 1:
                names = [
                    model.load_adapter(
                        os.path.join(args.restore_from, p), with_head=False
                    )
                    for p in sorted(os.listdir(args.restore_from))
                    if ("#" in p)
                ]
                model.load_adapter_fusion(
                    os.path.join(args.restore_from, "fusion_adapter_checkpoint"),
                    with_head=False,
                )
                fusion_adapter_name = Fuse(*names)
            else:
                debiasing_adapter_name = model.load_adapter(
                    os.path.join(args.restore_from, "debiasing_adapter_checkpoint"),
                    with_head=False,
                )
        elif args.add_task_adapter:
            logger.info(f"Loading task adapter from {args.restore_from}")
            task_adapter_name = model.load_adapter(
                os.path.join(args.restore_from, "task_adapter_checkpoint"),
                with_head=False,
            )

        # load the head separately
        logger.info("Loading classification head")
        model.load_head(os.path.join(args.restore_from, "head_checkpoint"))

        # set the active adapters
        logger.info("Setting active adapters (if any)")
        if args.add_debiasing_adapter and args.add_task_adapter:
            # make sure to stack fusion/debiasing adapter and task adapter
            if len(paths) > 1:
                logger.info(
                    "Fused Debiasing and task adapters found. Stacking both adapters and activating them"
                )
                model.set_active_adapters([fusion_adapter_name, task_adapter_name])
            else:
                logger.info(
                    "Debiasing and task adapters found. Stacking both adapters and activating them"
                )
                model.set_active_adapters([debiasing_adapter_name, task_adapter_name])
        elif args.add_debiasing_adapter:
            if len(paths) > 1:
                logger.info("Only fused debiasing adapter found. Activating it")
                model.set_active_adapters(fusion_adapter_name)
            else:
                logger.info("Only debiasing adapter found. Activating it")
                model.set_active_adapters(debiasing_adapter_name)
        elif args.add_task_adapter:
            logger.info("Only task adapter found. Activating it")
            model.set_active_adapters(task_adapter_name)

        # activate the head as well
        model.active_head = args.task_name

        logger.info(f"model: {model}")

        # as per the hacky implementation of InclusivityToolkit,
        # for MNLI, we get a DataFrame in which each element os a
        # probability distribution over the three classes (list) represented as a string.
        # We need to convert it to a list of floats to evaluate it
        df = eval_stsb_mnli(
            model=model,
            tokenizer=tokenizer,
            dirpath=args.test_set_path,
            block_size=args.block_size,
            dimension=args.dimension,
        )

        df.to_csv(os.path.join(args.restore_from, args.outfile), index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # in_dir: where the pretrained adapter is saved.
    # in case of fusion, it should be a ':' separated list of paths
    # where the debiasing adapters are saved
    # if a pre-trained fusion layer is available, then it should be given in the fusion_dir argument
    parser.add_argument("--dba_dir", type=str, default=None)
    parser.add_argument("--fusion_dir", type=str, default=None)
    parser.add_argument("--save_to", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--tokenizer_name", type=str, default="roberta-base")
    # block_size: max length of the input sequence
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--task_name", type=str, default="stsb")
    # add_debiasing_adapter: whether to add a debiasing adapter (or fusion)
    parser.add_argument("--add_debiasing_adapter", action="store_true")
    parser.add_argument("--add_task_adapter", action="store_true")
    parser.add_argument("--task_adapter_type", type=str, default="pfeiffer")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--run_id", type=str, default=None)
    # do not change this seed
    parser.add_argument("--seed", type=int, default=42)
    # Evalution arguments #
    # restore_from is the save_to you used during training
    # restore_from should only be used for loading trained/finetuned
    # models/adapters/heads during evaluation
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--restore_from", type=str, default=None)
    # test_set_path: path to the test set
    parser.add_argument("--test_set_path", type=str, default=None)
    # outfile: name of the file where the results will be saved (usually a .csv file)
    parser.add_argument("--dimension", type=str, default="gender")
    parser.add_argument("--outfile", type=str, default="similarity_scores.csv")
    args = parser.parse_args()

    # if a task adapter is not added, the ENTIRE model is finetuned end-to-end
    # if a task adapter is added, only the task adapter is finetuned
    # if a fusion layer + task adapter are present, both are trained (the constituent debiasing adapters are frozen)

    if args.run_id is None:
        args.run_id = f"{args.task_name}_{uuid.uuid4()}"

    if not args.do_eval:
        os.makedirs(args.save_to, exist_ok=True)
        with open(os.path.join(args.save_to, "ft_config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
