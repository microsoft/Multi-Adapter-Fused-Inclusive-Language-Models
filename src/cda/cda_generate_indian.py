import re
import os
import copy
import argparse
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from ADELE.CDA_wikipedia_bookcorpus import get_gender_word_pairs
import pdb


def merge_all_word_pairs():
    word_pairs = get_gender_word_pairs()
    word_pairs.append(["his", "her"])
    word_pairs.append(["her", "his"])
    # word_pairs.append(["himself", "herself"])
    # word_pairs.append(["herself", "himself"])
    word_pairs.append(["male", "female"])
    word_pairs.append(["female", "male"])
    word_pairs.append(["himself", "herself"])
    word_pairs.append(["herself", "himself"])

    return word_pairs


def cda_generate(
    output_filename, dataset, word_pairs, indianterms, max_lines, w_mode="w", **kwargs
):
    cda_type = kwargs.get("cda_type", "2-sided")
    with open(output_filename, w_mode) as f:
        counter = 0
        step = 0
        num_lines = 0
        for block in dataset["train"]:
            if num_lines >= max_lines:
                break
            block = block["text"]
            # put spaces in front of signs and replace new lines by dots
            block = (
                block.replace(",", " ,")
                .replace(":", " :")
                .replace(";", " ;")
                .replace("\n", " . ")
            )
            if (counter % 1000000) == 0:  # status update
                pass
                # print("wikipedia: ", counter, " / ", len(dataset_wikipedia["train"]))
            # split block to list of sentences
            s_list = re.split("[.?!]", block)
            counter += 1

            # alternating skip X (skip_sentences) sentences and take Y (take_sentences) sentences
            for i in range(len(s_list)):  # for each single sentence
                s = s_list[i].strip().lower()
                if len(s.split()) < 4:  # skip sentences with less than 3 tokens
                    continue
                # if step < args.skip_sentences: # skip first X sentences of block
                #     step += 1
                #     continue
                # elif step > args.block_size: # start new block
                #     step = 0
                #     continue
                # else: # take last Y sentences of block
                #     step += 1

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

                # Check if the sentence contains any indian term
                if any(word.lower() in indianterms for word in s_words):
                    # pdb.set_trace()
                    # if just a shorten original dataset is desired, add the original sentence and go on with the next sentence
                    # if cda_type == "original":
                    #     f.write(" ".join(s_words) + " . \n")
                    #     num_lines += 1
                    #     print(f"Number of Lines Written: {num_lines}",end='\r')
                    #     continue
                    # # if a shorten 2-sided CDA dataset is desired, add the original sentence
                    # if cda_type == "2-sided":
                    #     f.write(" ".join(s_words) + " . \n")
                    #     num_lines += 1

                    # check if the sentence contains words of the word list
                    s_words_cf = copy.deepcopy(s_words)
                    for j in range(len(s_words)):
                        for word_pair in word_pairs:
                            if (
                                s_words[j] == word_pair[0]
                            ):  # if there is a match, switch with the corresponding partner word
                                s_words_cf[j] = word_pair[1]
                                edit = True
                                break
                    # if there was a match, add the augmented sentence to new file
                    if edit:
                        f.write(" ".join(s_words) + " . \n")
                        f.write(" ".join(s_words_cf) + " . \n")
                        num_lines += 1
                        print(f"Number of Lines Written: {num_lines}", end="\r")
    print("...done\n\n")


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
    args = parser.parse_args()

    with open("aniket/indian_terms.txt", "r") as f:
        indianterms = set(f.read().split("\n"))

    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    word_pairs = merge_all_word_pairs()

    # 20200501.en is no longer in the datasets library, hence using the locally stored files
    # dataset_wikipedia = load_dataset("wikipedia", '20200501.en')
    dataset_wikipedia = load_from_disk(args.wikipedia_data_dir)

    cda_generate(
        args.output_file,
        dataset_wikipedia,
        word_pairs,
        indianterms,
        max_lines=40000,
        cda_type=args.cda_type,
        w_mode="w",
    )

    if args.bookcorpus:
        dataset_bc = load_dataset("bookcorpus")
        cda_generate(
            args.output_file,
            dataset_bc,
            word_pairs,
            indianterms,
            max_lines=40000,
            cda_type=args.cda_type,
            w_mode="a",
        )
