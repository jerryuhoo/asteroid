import pandas as pd

tag = "baseline"
# tag = "multiloss_with_psy"
# tag = "single_psy"
# tag = "multiloss_without_psy"
# tag = "finetune"
# tag = "multiloss_with_psy_cl"

results_from_file = pd.read_pickle(
    "exp/train_xumx_" + tag + "/EvaluateResults_musdb18_testdata/results.pandas"
)

# print(results_from_file)

# Filtering the DataFrame for the relevant targets
relevant_targets = ["vocals", "bass", "drums", "other"]
filtered_df = results_from_file[results_from_file["target"].isin(relevant_targets)]

# Grouping by 'metric', then calculating the median and mean for the scores
avg_scores = filtered_df.groupby("metric")["score"].agg(["median", "mean"]).reset_index()

# Setting the 'target' column to 'avg' for the average scores
avg_scores["target"] = "avg"

# Grouping by 'target' and 'metric', then calculating the median and mean for the scores
grouped_results_without_time = (
    results_from_file.groupby(["target", "metric"])["score"].agg(["median", "mean"]).reset_index()
)

# Combining the results with the average scores
combined_results = pd.concat([grouped_results_without_time, avg_scores])

# Sorting the results for better readability and formatting the results
combined_results.sort_values(by=["target", "metric"], inplace=True)
combined_results.rename(
    columns={
        "target": "Target",
        "metric": "Metric",
        "median": "Median Score",
        "mean": "Mean Score",
    },
    inplace=True,
)

# Rounding the scores to two decimal places
combined_results[["Median Score", "Mean Score"]] = combined_results[
    ["Median Score", "Mean Score"]
].round(2)

# Printing the formatted results
print(combined_results.head(20))
