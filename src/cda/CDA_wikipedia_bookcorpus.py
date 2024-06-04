from datasets import load_dataset
import re
import argparse

WORD_PAIR_DIR = "/home/t-assathe/ModularRAI/src/wordpairs/"
# checks if list already contains the word pair
def is_pair_in_list(all_pairs, pair):
    for p in all_pairs:
        if (p[0] == pair[0]) and p[1] == pair[1]:
            return True
    return False


# returns word list of noun pairs of Zhao et al. and 100 self-created name pairs
def get_gender_word_pairs():
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

    # file_wordlist = open(f'{WORD_PAIR_DIR}/indian_name_pairs.txt', 'r')
    # lines_wordlist = file_wordlist.readlines()
    # for line in lines_wordlist:
    #     word_pair = line.split()
    #     if not is_pair_in_list(word_pairs, word_pair):
    #         word_pairs.append(word_pair)
    #         word_pairs.append([word_pair[1], word_pair[0]]) # both 'dircetions' needed: (male, female) and (female, male)

    # Word pairs extracted using wikidata and gpt-3
    # file_wordist = open(f"{WORD_PAIR_DIR}/gpt_generated_gendered_pairs_v2.txt", 'r')
    file_wordist = open(f"{WORD_PAIR_DIR}/gpt_generated_gendered_pairs_v2.txt", "r")
    lines_wordlist = file_wordist.readlines()
    for line in lines_wordlist:
        word_pair = line.split("\n")[0]
        word_pair = word_pair.replace("(", "").replace(")", "").split(",")
        word_pair = list(map(lambda x: x.strip().split()[0], word_pair))
        if not is_pair_in_list(word_pairs, word_pair):
            word_pairs.append(word_pair)
            word_pairs.append([word_pair[1], word_pair[0]])

    return word_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="The output for the counterfactual augmented dataset.",
    )
    parser.add_argument(
        "--cda_type", type=str, required=True, help="original or 1-sided or 2-sided"
    )
    parser.add_argument(
        "--skip_sentences",
        type=int,
        required=True,
        help="To get just a fraction of the wikipedia-bookcorpus dataset: alternating 'skip X sentences' and 'take Y sentence'",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        required=True,
        help="number of sentences treated as a block: sum of 'skip X senteces' and 'take Y sentences'",
    )
    args = parser.parse_args()

    print("Get gender word pairs...")
    word_pairs = get_gender_word_pairs()
    # do some adjustments
    word_pairs.append(["his", "her"])
    word_pairs.append(["her", "his"])
    word_pairs.append(["himself", "herself"])
    word_pairs.append(["herself", "himself"])
    word_pairs.append(["male", "female"])
    word_pairs.append(["female", "male"])
    word_pairs.append(["himself", "herself"])
    word_pairs.append(["herself", "himself"])
    print("...done\n\n")

    # Load datasets of wikipedia and bookcorpus
    print("Load bookcorpus...")
    dataset_bookcorpus = load_dataset("bookcorpus")
    print("...done\n\n")
    print("Load wikipedia...")
    dataset_wikipedia = load_dataset("wikipedia", "20200501.en")
    print("...done\n\n")

    # create the output text file
    f = open(args.output_file, "w+")
    f.close()
    # open the output text file to append sentence by sentence
    with open(args.output_file, "a", encoding="utf-8") as f:

        # ---------------------------------------------BOOKCORPUS---------------------------------------------
        print("Create list of sentences of bookcorpus...")
        counter = 0
        step = 0
        # bookcorpus: extract data block by block
        for block in dataset_bookcorpus["train"]:
            block = block["text"]
            # put spaces in front of signs and replace new lines by dots
            block = (
                block.replace(",", " ,")
                .replace(":", " :")
                .replace(";", " ;")
                .replace("\n", " . ")
            )
            if (counter % 10000000) == 0:  # status update
                print("bookcorpus: ", counter, " / ", len(dataset_bookcorpus["train"]))
            # split block to list of sentences
            s_list = re.split("[.?!]", block)
            counter += 1

            # alternating skip X (skip_sentences) sentences and take Y (take_sentences) sentences
            for i in range(len(s_list)):  # for each single sentence
                s = s_list[i].strip().lower()
                if len(s.split()) < 4:  # skip sentences with less than 4 tokens
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
                if args.cda_type == "original":
                    f.write(" ".join(s_words) + " . \n")
                    continue
                # if a shorten 2-sided CDA dataset is desired, add the original sentence
                if args.cda_type == "2-sided":
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

        # ---------------------------------------------WIKIPEDIA---------------------------------------------
        print("Create list of sentences of wikipedia...")
        counter = 0
        step = 0
        # wikipedia: extract data block by block
        for block in dataset_wikipedia["train"]:
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
                if args.cda_type == "original":
                    f.write(" ".join(s_words) + " . \n")
                    continue
                # if a shorten 2-sided CDA dataset is desired, add the original sentence
                if args.cda_type == "2-sided":
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
