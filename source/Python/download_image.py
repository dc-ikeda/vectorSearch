import os
os.environ["HF_DATASETS_CACHE"] = "D:\\Python\\HuggingFaceCache_image"


from datasets import load_dataset

dataset = load_dataset("recruit-jp/japanese-image-classification-evaluation-dataset")

output_path = "D:\\Python\\image\\recruit-jp.parquet"
dataset["train"].to_parquet(output_path)