# Introduction

Language Models such as GPT-3 and Turing's Language Representation Models form the backbone of several technologies within Microsoft and beyond. Many of these are user-facing , while others are used in critical decision making. These models are extremely powerful but are difficult to debug and interpret. Inclusivity can be a challenge for these models in the following two ways:

- Models can be offensive or propagate stereotypes

- Models may not provide the same quality of service to all users

Taking inspiration from Turing's challenge for Inclusivity, we have created a toolkit that brings together various internal and open source datasets available for evaluating models across different dimensions.

Our toolkit can be used to test foundation models on intrinsic evaluation metrics of fairness, which we then plan to extend to extrinsic metrics for task-specific evaluation. 

# Setup

We recommend creating a virtual environment before installing the package (optional):

```shell
$ [sudo] pip install virtualenv
$ virtualenv -p python3 toolkitenv
$ source toolkitenv/bin/activate
```

Install the toolkit

```shell
$ cd InclusivityToolkit
$ pip install --editable ./
```

# Example Use Cases

## Stereoset ([Nadeem et al. 2021](https://aclanthology.org/2021.acl-long.416/))

```python
from inclusivity_toolkit import eval_stereoset

# For encoder only model like bert:
from transformers import AutoTokenizer, AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(eval_stereoset(model, tokenizer model_type="encoder-only"))

# For encoder only model like gpt2:
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(eval_stereoset(model, tokenizer model_type="decoder-only"))
```

## CrowS-Pairs ([Nangia et al. 2020](https://aclanthology.org/2020.emnlp-main.154))

```python
from inclusivity_toolkit import eval_crows

# CrowS only supports encoder-only models
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(eval_crows(model, tokenizer))
```

## Extrinsic Evaluation (STS-B and MNLI)

```python

from inclusivity_toolkit import eval_stsb_mnli

# Extrinsic evaluations like STS-B and MNLI require the model to have
# a task specific head. Generally, the head is trained by fine-tuning on
# the original GLUE STS-B and MNLI train set.

# Note that, this method return a pandas DataFrame with the semantic similarity
# or entailment probabilities (entailment, neutral or contradiction) for each sentence-pair
# The dimension, such as gender, race, religion can be specified as an argument

dimension = "gender" # race, religion

df = eval_stsb_mnli(
    model=model, # here, the model is expected to have a task specific head
    tokenizer=tokenizer, # tokenizer corresponding to the model being evaluated
    dimension=dimension
    dirpath=dirpath # path to the testset, can be hard-coded
)

print(df)
```

## Credits
Apart from the authors, the following people had contributed to the development of the library:

- Kritika Ramesh
- Rishav Hada 
- Shrey Pandit 
- Aniket Vashishtha
- Nidhi Kulkarni
- Ananya Saxena
- Pamir Gogoi
- Suril Mehta
- Sapna
