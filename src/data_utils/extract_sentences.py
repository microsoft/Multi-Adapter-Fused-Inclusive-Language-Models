import re
import os
import csv
from glob import glob
import argparse
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
import pdb

WIKIDATA_DIR = "/home/t-kabirahuja/work/repos/ModularRAI/data/wikipedia/20200501.en.hf"
TERMS_DIR = "/home/t-kabirahuja/work/repos/ModularRAI/data/readable_bias_entity_en_1"


def get_terms_from_file(filename):
    terms = []
    with open(filename) as f:
        for line in f:
            terms.append(line.split()[-1].lower())

    return terms


def load_terms(dimension="religion"):
    filenames = glob(f"{TERMS_DIR}/readable_{dimension}*.txt")
    term_dict = {}
    term2cat = {}
    for filename in filenames:
        category = "_".join(filename.split("/")[-1].split("_")[1:-1])
        term_dict[category] = terms
        term2cat[term] = category
    return term2cat


def extract_sents_with_terms(
    dataset, term2cat, output_filename, w_mode="a", debug=True
):
    with open(output_filename, w_mode) as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        for block_id, block in tqdm(enumerate(dataset["train"])):
            if debug and block_id == 1000:
                break
            block = block["text"]
            # put spaces in front of signs and replace new lines by dots
            block = (
                block.replace(",", " ,")
                .replace(":", " :")
                .replace(";", " ;")
                .replace("\n", " . ")
            )
            s_list = re.split("[.?!]", block)
            counter += 1
            for i in range(len(s_list)):  # for each single sentence
                s = s_list[i].strip().lower()
                if len(s.split()) < 4:  # skip sentences with less than 3 tokens
                    continue
                labels = []
                for term, cat in term2cat.items():
                    if term in s:
                        labels.append(f"{cat}:{term}")
                if labels == []:
                    continue

                writer.writerow([s, ",".join(labels)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="The output for the counterfactual augmented dataset.",
    )

    parser.add_argument("-d", "--dimension", default="religion", type=str)
    parser.add_argument(
        "--bookcorpus", action="store_true", help="Whether to also append bookcorpus"
    )
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    term2cat = load_term(dimension=args.dimension)
    pdb.set_trace()
    dataset_wikipedia = load_from_disk(WIKIDATA_DIR)
    extract_sents_with_terms(
        dataset_wikipedia,
        term2cat,
        args.output_file,
    )

    if args.bookcorpus:
        dataset_bc = load_dataset("bookcorpus")
        extract_sents_with_terms(dataset_bc, term2cat, args.output_file)
