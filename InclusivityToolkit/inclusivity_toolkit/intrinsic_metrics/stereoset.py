import os
from inclusivity_toolkit.evaluator_utils.stereoset import (
    evaluate_intrasentence,
    evaluate_intrasentence_dec_only,
)
from inclusivity_toolkit.evaluators.StereoSet.code.evaluation import ScoreEvaluator

TOOLKIT_DIR = os.path.split(os.path.dirname(__file__))[0]
STEREOSET_DATA_DIR = os.path.join(TOOLKIT_DIR, "evaluators/StereoSet/data/")


def eval_stereoset_encoder_only(model, tokenizer, dimension=None, **kwargs):
    input_file = (
        f"{STEREOSET_DATA_DIR}/dev.json"
        if dimension is None
        else f"{STEREOSET_DATA_DIR}/dev_{dimension}.json"
    )
    preds = evaluate_intrasentence(
        model,
        tokenizer,
        input_file=input_file,
        batch_size=kwargs.get("batch_size", 32),
        max_seq_length=kwargs.get("max_seq_length", 128),
        device=model.device,
    )
    score_evaluator = ScoreEvaluator(
        gold_file_path=input_file, predictions={"intrasentence": preds}
    )
    return score_evaluator.results


def eval_stereoset_decoder_only(model, tokenizer, dimension=None, **kwargs):
    input_file = (
        f"{STEREOSET_DATA_DIR}/dev.json"
        if dimension is None
        else f"{STEREOSET_DATA_DIR}/dev_{dimension}.json"
    )
    preds = evaluate_intrasentence_dec_only(
        model,
        tokenizer,
        input_file=input_file,
        batch_size=kwargs.get("batch_size", 1),
        max_seq_length=kwargs.get("max_seq_length", 128),
        device=model.device,
    )
    score_evaluator = ScoreEvaluator(
        gold_file_path=input_file, predictions={"intrasentence": preds}
    )
    return score_evaluator.results


def eval_stereoset(
    model, tokenizer, dimension=None, model_type="encoder-only", **kwargs
):

    if model_type == "encoder-only":
        return eval_stereoset_encoder_only(
            model, tokenizer, dimension=dimension, **kwargs
        )

    if model_type == "decoder-only":
        return eval_stereoset_decoder_only(
            model, tokenizer, dimension=dimension, **kwargs
        )
