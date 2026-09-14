import pandas as pd
from datasets import load_dataset
from collections import Counter

ds = load_dataset("ashraq/fashion-product-images-small")

print(ds)
print(ds["train"].features)
print(ds["train"][0])



article_types = Counter(ds["train"]["articleType"])

print(f"Total de imagens: {len(ds['train'])}")
print(f"Total de articleTypes: {len(article_types)}")

for article_type, count in article_types.most_common():
    print(f"{article_type:<30} {count:>5}")



df = ds["train"].to_pandas()

article_summary = (
    df.groupby(["masterCategory", "subCategory", "articleType"])
      .size()
      .reset_index(name="count")
      .sort_values("count", ascending=False)
)

article_summary.head(50)

article_summary.to_csv(
    "article_type_summary.csv",
    index=False
)

