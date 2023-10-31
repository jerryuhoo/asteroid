import pandas as pd

# tag = "baseline"
tag = "multiloss_with_psy"

results_from_file = pd.read_pickle(
    "exp/train_xumx_" + tag + "/EvaluateResults_musdb18_testdata/results.pandas"
)

avg_scores = results_from_file.groupby("metric")["score"].mean()
ordered_metrics = ["SDR", "SIR", "ISR", "SAR"]
avg_scores = avg_scores.loc[ordered_metrics]
print(avg_scores)
