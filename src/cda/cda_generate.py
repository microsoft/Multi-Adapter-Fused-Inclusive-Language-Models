"""
python cda_generate.py --output_file  ~/project/data/wikipedia/wiki_cda/race/raw.txt --wikipedia_data_dir  ~/project/data/wikipedia/20200501.en.hf/ --bias_type race
"""

import re
import os
import argparse
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from CDA_wikipedia_bookcorpus import get_gender_word_pairs
import numpy as np
import ipdb

WORD_PAIR_DIR = "/home/t-assathe/ModularRAI/src/wordpairs/"


def get_unique_cf_pairs(inList):
    inList = [ele[0] + "\t" + ele[1] for ele in inList]
    inList = np.unique(inList)
    inList = [ele.split("\t") for ele in inList]
    return inList


# checks if list already contains the word pair
def is_pair_in_list(all_pairs, pair):
    for p in all_pairs:
        if (p[0] == pair[0]) and p[1] == pair[1]:
            return True
    return False


def adele_word_pairs():
    word_pairs = []
    # https://github.com/uclanlp/corefBias/blob/master/WinoBias/wino/generalized_swaps.txt
    # creates list with word pairs --> [ [pair1[0], pair1[1]] , [pair2[0], pair2[1]] , ... ]
    file_wordlist = open(f"{WORD_PAIR_DIR}/./cda_word_pairs_gender.txt", "r")
    lines_wordlist = file_wordlist.readlines()
    for line in lines_wordlist:
        word_pair = line.split()
        word_pairs.append(word_pair)

    # https://github.com/uclanlp/corefBias/blob/master/WinoBias/wino/extra_gendered_words.txt
    # appends additional word pairs from extra file
    file_wordlist = open(f"{WORD_PAIR_DIR}/cda_word_pairs_gender_extra.txt", "r")
    lines_wordlist = file_wordlist.readlines()
    for line in lines_wordlist:
        word_pair = line.split()
        if not is_pair_in_list(word_pairs, word_pair):
            word_pairs.append(word_pair)
            word_pairs.append(
                [word_pair[1], word_pair[0]]
            )  # both 'dircetions' needed: (male, female) and (female, male)

    # https://www.ssa.gov/oact/babynames/limits.html
    # gets the top 100 names of 2019 for boys and girls and appends the pairs (male, female) and (female, male) to the word pair list
    file_wordlist = open(f"{WORD_PAIR_DIR}/cda_word_pairs_names.txt", "r")
    lines_wordlist = file_wordlist.readlines()
    for line in lines_wordlist:
        word_pair = line.split()
        if not is_pair_in_list(word_pairs, word_pair):
            word_pairs.append(word_pair)

    return word_pairs


def merge_all_word_pairs():
    word_pairs = get_gender_word_pairs()
    word_pairs.append(["his", "her"])
    word_pairs.append(["her", "his"])
    word_pairs.append(["himself", "herself"])
    word_pairs.append(["herself", "himself"])
    word_pairs.append(["male", "female"])
    word_pairs.append(["female", "male"])
    word_pairs.append(["himself", "herself"])
    word_pairs.append(["herself", "himself"])
    return word_pairs


def read_file(filename, delimiter=","):
    # Word pairs extracted using wikidata and gpt-3
    print("Reading...", filename)
    word_pairs = []
    file_wordist = open(filename, "r")
    lines_wordlist = file_wordist.readlines()
    file_wordist.close()
    for line in lines_wordlist:
        try:
            word_pair = line.split("\n")[0]
            word_pair = word_pair.replace("(", "").replace(")", "").split(delimiter)
            word_pair = list(map(lambda x: x.strip().split()[0], word_pair))
            if not is_pair_in_list(word_pairs, word_pair):
                word_pairs.append(word_pair)
                word_pairs.append([word_pair[1], word_pair[0]])
        except:
            ipdb.set_trace()
    return word_pairs


def get_race_example_shiva():
    race_triple = [
        "(black, caucasian, asian)",
        "(african, caucasian, asian)",
        "(black, white, asian)",
    ]
    race_pairs = []
    for race in race_triple:
        race = race.strip("(").strip(")").split(", ")
        j = 0
        for ele in race:
            j += 1
            for i in range(j, 3):
                race_pairs.append([ele, race[i]])
    race_pairs = get_unique_cf_pairs(race_pairs)
    return race_pairs


def get_religion_example_shiva():
    religion_triple = [
        "(jewish, christian, muslim)",
        "(jews, christians, muslims)",
        "(torah, bible, quran)",
        "(synagogue, church, mosque)",
        "(rabbi, priest, imam)",
        "(judaism, christianity, islam)",
    ]
    religion_pairs = []
    for religion in religion_triple:
        religion = religion.strip("(").strip(")").split(", ")
        j = 0
        for ele in religion:
            j += 1
            for i in range(j, 3):
                religion_pairs.append([ele, religion[i]])
    religion_pairs = get_unique_cf_pairs(religion_pairs)
    return religion_pairs


def merge_all_word_pairs_fromfile(bias_type):
    word_pairs = []
    path = "/home/t-jainprachi/project/data/readable_bias_entity_en_1/cf/pre-corrected-prompts/"
    if bias_type == "race":
        word_pairs = get_race_example_shiva()
        word_pairs += read_file(path + "race_sub/" + "gpt_generated_race_pairs.txt")
    elif bias_type == "gender":
        word_pairs = merge_all_word_pairs()
    elif bias_type == "gender_adele":
        word_pairs = adele_word_pairs()
    elif bias_type == "profession":
        path = "/home/t-jainprachi/project/data/readable_bias_entity_en_1/cf/"
        word_pairs = read_file(
            path + "profession_sub/" + "gpt_generated_profession_pairs.txt"
        )
        path = "/home/t-jainprachi/project/code/ModularRAI/ADELE/datasets/wordpairs/"
        word_pairs += read_file(
            path + "cda_word_pairs_gender_extra.txt", delimiter="\t"
        )
        word_pairs += read_file(path + "disco_word_pairs_nouns.txt", delimiter=" ")
    elif bias_type == "religion":
        word_pairs = get_religion_example_shiva()
        word_pairs += read_file(
            path + "religion_sub/" + "gpt_generated_religion_pairs.txt"
        )

    return get_unique_cf_pairs(word_pairs)


def cda_generate(
    output_filename,
    dataset,
    word_pairs,
    w_mode="w",
    skip_sentences=0,
    block_size=512,
    **kwargs,
):
    cda_type = kwargs.get("cda_type", "2-sided")
    f = open(output_filename, w_mode)

    counter = 0
    step = 0
    for block in tqdm(dataset["train"]):
        block = block["text"]
        # put spaces in front of signs and replace new lines by dots
        block = (
            block.replace(",", " ,")
            .replace(":", " :")
            .replace(";", " ;")
            .replace("\n", " . ")
        )
        if (counter % 1000000) == 0:  # status update
            print("wikipedia: ", counter, " / ", len(dataset_wikipedia["train"]))
        # split block to list of sentences
        s_list = re.split("[.?!]", block)
        counter += 1

        # alternating skip X (skip_sentences) sentences and take Y (take_sentences) sentences
        for i in range(len(s_list)):  # for each single sentence
            s = s_list[i].strip().lower()
            if len(s.split()) < 4:  # skip sentences with less than 3 tokens
                continue
            if step < args.skip_sentences:  # skip first X sentences of block
                step += 1
                continue
            elif step > args.block_size:  # start new block
                step = 0
                continue
            else:  # take last Y sentences of block
                step += 1

            # only executed if sentence belongs to last Y sentences of block
            edit = False
            # split sentence to words and eliminate whitespaces around the words
            s_words = s.split()
            s_words[:] = [x.strip() for x in s_words if x]
            # there are a lot of wikipeda sentences starting with category, remove the word category
            if s_words[0] == "category":
                s_words = s_words[1:]
                s_words[0] = s_words[0][1:]
            if len(s_words) < 4:
                continue

            # if just a shorten original dataset is desired, add the original sentence and go on with the next sentence
            if cda_type == "original":
                f.write(" ".join(s_words) + " . \n")
                continue
            # if a shorten 2-sided CDA dataset is desired, add the original sentence
            if cda_type == "2-sided":
                f.write(" ".join(s_words) + " . \n")

            # check if the sentence contains words of the word list
            for j in range(len(s_words)):
                for word_pair in word_pairs:
                    if (
                        s_words[j] == word_pair[0]
                    ):  # if there is a match, switch with the corresponding partner word
                        s_words[j] = word_pair[1]
                        edit = True
                        break
            # if there was a match, add the augmented sentence to new file
            if edit:
                f.write(" ".join(s_words) + " . \n")
    print("...done\n\n")
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="The output for the counterfactual augmented dataset.",
    )
    parser.add_argument(
        "--cda_type",
        type=str,
        default="2-sided",
        choices=["2-sided", "1-side", "original"],
        help="original or 1-sided or 2-sided",
    )
    parser.add_argument(
        "--wikipedia_data_dir",
        type=str,
        default="data/wikipedia/20200501.en.hf",
        help="Local path where wikipedia data is located",
    )
    parser.add_argument(
        "--bookcorpus", action="store_true", help="Whether to also append bookcorpus"
    )
    parser.add_argument(
        "--skip_sentences",
        type=int,
        default=0,
        help="To get just a fraction of the wikipedia-bookcorpus dataset: alternating 'skip X sentences' and 'take Y sentence'",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=512,
        help="number of sentences treated as a block: sum of 'skip X senteces' and 'take Y sentences'",
    )
    parser.add_argument(
        "--bias_type",
        type=str,
        choices=[
            "race",
            "gender",
            "profession",
            "religion",
            "gender_adele",
            "race_shiva",
            "religion_shiva",
        ],
        help="bias for which data is to be generated",
    )

    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    word_pairs = merge_all_word_pairs_fromfile(args.bias_type)

    # 20200501.en is no longer in the datasets library, hence using the locally stored files
    # dataset_wikipedia = load_dataset("wikipedia", '20200501.en')
    dataset_wikipedia = load_from_disk(args.wikipedia_data_dir)

    cda_generate(
        args.output_file,
        dataset_wikipedia,
        word_pairs,
        cda_type=args.cda_type,
        w_mode="w",
        skip_sentences=args.skip_sentences,
        block_size=args.block_size,
    )
    if args.bookcorpus:
        dataset_bc = load_dataset("bookcorpus")
        cda_generate(
            args.output_file, dataset_bc, word_pairs, cda_type=args.cda_type, w_mode="a"
        )
