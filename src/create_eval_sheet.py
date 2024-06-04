from itertools import combinations
from glob import glob
from tqdm import tqdm
import pandas as pd
import json

root_dir = "/home/t-assathe/local_ckpts_stacked_XLMR/fusion_stsb/"

dimensions = ["gender", "race", "religion"]

models = glob(f"{root_dir}/*")

results = pd.DataFrame(columns=["model_name", "accuracy"] + dimensions)
for model in tqdm(models):
    dim_to_df = {}
    for dim in dimensions:
        print(model, dim)
        df = pd.read_csv(f"{model}/raw_dim={dim}.csv")
        df = df.drop("Unnamed: 0", axis=1)
        comb_columns = list(combinations(list(df.columns), 2))
        average = 0
        for column_x, column_y in comb_columns:
            diff = abs(df[column_x] - df[column_y])
            df[f"{column_x}::{column_y}"] = diff
            average += diff
        df["overall_delta"] = average / len(comb_columns)
        print(df.shape, df["overall_delta"].mean(), len(comb_columns))
        dim_to_df[dim] = df
    new_dict = {"model_name": model}
    with open(f"{model}/eval_results.json") as f:
        new_dict["accuracy"] = json.load(f)["eval_accuracy"]
    with pd.ExcelWriter(f"{model}/all_dims_processed.xlsx") as writer:
        for dim, df in dim_to_df.items():
            res = df["overall_delta"].mean()
            df.to_excel(writer, sheet_name=dim)
            new_dict[dim] = res
    results = pd.concat([results, pd.DataFrame([new_dict])], ignore_index=True)
results = results.set_index("model_name")
results = results.sort_values("model_name")
results.to_csv("results_extrinsic.csv")
