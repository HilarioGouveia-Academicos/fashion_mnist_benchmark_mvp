import pandas as pd

df = pd.read_csv(
    "artifacts/domain_shift_v2/external_predictions.csv"
)

print(
    df.groupby(
        ["model", "predicted_name"]
    )
    .size()
    .unstack(fill_value=0)
)