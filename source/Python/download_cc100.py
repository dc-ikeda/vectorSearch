import os
os.environ["HF_DATASETS_CACHE"] = "D:\\Python\\HuggingFaceCache"

from datasets import load_dataset


dataset = load_dataset("cc100", lang="ja", trust_remote_code=True)
#getlist = dataset["train"].select(range(100000))

output_path = "D:\\Python\\cc100_Parquet\\cc100-ja_sharded2.parquet"
#getlist.to_parquet(output_path)
dataset["train"].to_parquet(output_path)