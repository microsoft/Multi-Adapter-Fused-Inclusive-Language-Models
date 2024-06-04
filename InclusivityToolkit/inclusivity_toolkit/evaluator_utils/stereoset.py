import os
from collections import defaultdict

import collections
import collections.abc

for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import inclusivity_toolkit.evaluators.StereoSet.code.dataloader as dataloader


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_intrasentence(
    model, tokenizer, input_file, batch_size=32, max_seq_length=128, device="cuda"
):

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.eval()

    print()
    print(f"Evaluating bias on intrasentence tasks...")
    print(f"Number of parameters: {count_parameters(model):,}")

    pad_to_max_length = True if batch_size > 1 else False
    dataset = dataloader.IntrasentenceLoader(
        tokenizer,
        max_seq_length=max_seq_length,
        pad_to_max_length=pad_to_max_length,
        input_file=input_file,
    )

    loader = DataLoader(dataset, batch_size=batch_size)
    word_probabilities = defaultdict(list)

    # calculate the logits for each prediction
    for sentence_id, next_token, input_ids, attention_mask, token_type_ids in tqdm(
        loader, total=len(loader)
    ):
        # start by converting everything to a tensor
        input_ids = torch.stack(input_ids).to(device).transpose(0, 1)
        attention_mask = torch.stack(attention_mask).to(device).transpose(0, 1)
        next_token = next_token.to(device)
        token_type_ids = torch.stack(token_type_ids).to(device).transpose(0, 1)

        MASK_TOKEN = tokenizer.mask_token
        MASK_TOKEN_IDX = tokenizer.encode(MASK_TOKEN, add_special_tokens=False)[0]
        mask_idxs = input_ids == MASK_TOKEN_IDX

        # get the probabilities
        try:
            output = model(
                input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            )[0].softmax(dim=-1)
        except:
            output = model(input_ids, attention_mask=attention_mask)[0].softmax(dim=-1)

        output = output[mask_idxs]

        output = output.index_select(1, next_token).diag()
        for idx, item in enumerate(output):
            word_probabilities[sentence_id[idx]].append(item.item())

    # now reconcile the probabilities into sentences
    sentence_probabilties = []
    for k, v in word_probabilities.items():
        pred = {}
        pred["id"] = k
        # score = np.sum([np.log2(i) for i in v]) + np.log2(len(v))
        score = np.mean(v)
        pred["score"] = score
        sentence_probabilties.append(pred)

    return sentence_probabilties


def evaluate_intrasentence_dec_only(
    model, tokenizer, input_file, batch_size=32, max_seq_length=128, device="cuda"
):

    UNCONDITIONAL_START_TOKEN = "<|endoftext|>"

    print()
    print(f"Evaluating bias on intrasentence tasks...")
    print(f"Number of parameters: {count_parameters(model):,}")

    model.eval()

    start_token = (
        torch.tensor(tokenizer.encode(UNCONDITIONAL_START_TOKEN))
        .to(device)
        .unsqueeze(0)
    )
    initial_token_probabilities = model(start_token)
    initial_token_probabilities = torch.softmax(initial_token_probabilities[0], dim=-1)

    # ensure that our batch size is 1, and that our initial token isn't split into subwords.
    assert initial_token_probabilities.shape[0] == 1
    assert initial_token_probabilities.shape[1] == 1

    filename = os.path.abspath(input_file)
    stereoset_dataloader = dataloader.StereoSet(filename)

    clusters = stereoset_dataloader.get_intrasentence_examples()
    predictions = []
    for cluster in tqdm(clusters):
        for sentence in cluster.sentences:
            probabilities = {}
            tokens = tokenizer.encode(sentence.sentence)
            joint_sentence_probability = [
                initial_token_probabilities[0, 0, tokens[0]].item()
            ]
            tokens_tensor = torch.tensor(tokens).to(device).unsqueeze(0)
            output = torch.softmax(model(tokens_tensor)[0], dim=-1)
            for idx in range(1, len(tokens)):
                joint_sentence_probability.append(
                    output[0, idx - 1, tokens[idx]].item()
                )

            # ensure that we have a probability on every token
            assert len(tokens) == len(joint_sentence_probability)

            score = np.sum([np.log2(i) for i in joint_sentence_probability])
            score /= len(joint_sentence_probability)
            score = np.power(2, score)

            probabilities["id"] = sentence.ID
            probabilities["score"] = score

            predictions.append(probabilities)

    return predictions


def process_job(batch, model, pretrained_class):
    input_ids, token_type_ids, sentence_id = batch
    outputs = model(input_ids, token_type_ids=token_type_ids)
    if type(outputs) == tuple:
        outputs = outputs[0]
    outputs = torch.softmax(outputs, dim=1)

    pid = sentence_id[0]
    # if "bert"==self.PRETRAINED_CLASS[:4]:
    if "bert" in pretrained_class:
        pscore = outputs[0, 0].item()
    else:
        pscore = outputs[0, 1].item()
    return (pid, pscore)


# def evaluate_intersentence(
#     model, tokenizer, input_file, batch_size=1, max_seq_length=128, device="cuda"
# ):
#     PRETRAINED_CLASS = model.config._name_or_path
#     print()
#     print(f"Evaluating bias on intersentence tasks...")

#     print(f"Number of parameters: {count_parameters(model):,}")
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     model = torch.nn.DataParallel(model)
#     model.eval()
#     dataset_args = AttrDict(
#         {
#             "input_file": input_file,
#             "max_seq_length": max_seq_length,
#             "batch_size": batch_size,
#         }
#     )
#     dataset = IntersentenceDataset(tokenizer, dataset_args)
#     # TODO: test this on larger batch sizes.
#     assert batch_size == 1
#     dataloader = DataLoader(dataset, shuffle=True, num_workers=0)
#     no_cuda = "cuda" not in device
#     if no_cuda:
#         n_cpus = cpu_count()
#         print(f"Using {n_cpus} cpus!")
#         predictions = Parallel(n_jobs=n_cpus, backend="multiprocessing")(
#             delayed(process_job)(batch, model, PRETRAINED_CLASS)
#             for batch in tqdm(dataloader, total=len(dataloader))
#         )
#     else:
#         predictions = []

#         for batch_num, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
#             input_ids, token_type_ids, attention_mask, sentence_id = batch
#             input_ids = input_ids.to(device)
#             token_type_ids = token_type_ids.to(device)
#             attention_mask = attention_mask.to(device)
#             outputs = model(input_ids, token_type_ids=token_type_ids)
#             if type(outputs) == tuple:
#                 outputs = outputs[0]
#             if isinstance(outputs, CausalLMOutputWithCrossAttentions):
#                 outputs = outputs.logits
#             import pdb

#             pdb.set_trace()
#             outputs = torch.softmax(outputs, dim=1)

#             for idx in range(input_ids.shape[0]):
#                 probabilities = {}
#                 probabilities["id"] = sentence_id[idx]
#                 if "bert" == PRETRAINED_CLASS[:4] or "roberta-base" == PRETRAINED_CLASS:
#                     probabilities["score"] = outputs[idx, 0].item()
#                 else:
#                     probabilities["score"] = outputs[idx, 1].item()
#                 predictions.append(probabilities)

#     return predictions
