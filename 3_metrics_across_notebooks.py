import pandas as pd
import glob

def compute_avg_and_em_metrics(folder_name):
    metrics_all = []
    em_min, em_max, em_baseline, em_crabs, em_a1, em_a2 = 0, 0, 0, 0, 0, 0
    count = 0
    for file_path in sorted(glob.glob(f"data/outputs/{folder_name}/*.csv")):
        print(file_path)
        # total number of notebooks
        count = count + 1
        # Load the metrics and append to the list
        metrics_per_notebook = pd.read_csv(file_path, index_col=0)
        metrics_all.append(metrics_per_notebook)
        # Count extract matched notebooks
        if metrics_per_notebook.loc["f1", "MinIOParser"] == 1:
            em_min = em_min + 1
        if metrics_per_notebook.loc["f1", "MaxIOParser"] == 1:
            em_max = em_max + 1
        if metrics_per_notebook.loc["f1", "Baseline"] == 1:
            em_baseline = em_baseline + 1
        if metrics_per_notebook.loc["f1", "CRABS"] == 1:
            em_crabs = em_crabs + 1
        if metrics_per_notebook.loc["f1", "A1"] == 1:
            em_a1 = em_a1 + 1
        if metrics_per_notebook.loc["f1", "A2"] == 1:
            em_a2 = em_a2 + 1

    metrics_all_avg = pd.concat(metrics_all).groupby(level=0).agg(['mean', 'std']).apply(lambda x: x.xs('mean', level=1).map('{:.4f}'.format) + '(' + x.xs('std', level=1).map('{:.4f}'.format) + ')', axis=1).reset_index()
    metrics_all_avg.loc[len(metrics_all_avg)] = ["em", em_min/count, em_max/count, em_baseline/count, em_crabs/count, em_a1/count, em_a2/count]
    return metrics_all_avg

for folder_name in ["informationflowgraph", "cellexecutiondependencygraph"]:
    metrics = compute_avg_and_em_metrics(folder_name)
    metrics.to_csv(f"data/outputs/{folder_name}.csv", index=False)
