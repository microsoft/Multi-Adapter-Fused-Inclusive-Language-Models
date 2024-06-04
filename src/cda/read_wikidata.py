from tqdm import tqdm
from datasets import load_from_disk
import pdb

if __name__ == "__main__":

    # Path containing the wikidata
    data_dir = "data/wikipedia/20200501.en.hf"

    # Loading the dataset
    dataset_wikipedia = load_from_disk(data_dir)["train"]

    # Iterating through the dataset
    for block in tqdm(dataset_wikipedia):
        title = block["title"]
        text = block["text"]
        print(title)
        print(text)
        # Breaking for demonstration but can go about according to the need
        break
