import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def statsTable(averageResultsMatrix, compName):
    metric_names = [row[0] for row in averageResultsMatrix[0]]
    n_metrics = len(metric_names)

    # Extract values into arrays
    orig_values = np.array([[row[m][1] for m in range(n_metrics)]
                            for row in averageResultsMatrix])
    comp_values = np.array([[row[m][2] for m in range(n_metrics)]
                            for row in averageResultsMatrix])

    # Compute statistics
    mean_orig = orig_values.mean(axis=0)
    std_orig  = orig_values.std(axis=0)
    mean_comp = comp_values.mean(axis=0)
    std_comp  = comp_values.std(axis=0)

    # Column headers
    orig_col = [f"{mean_orig[i]:.4f} ± {std_orig[i]:.4f}" for i in range(n_metrics)]
    comp_col = [f"{mean_comp[i]:.4f} ± {std_comp[i]:.4f}" for i in range(n_metrics)]

    # Build table
    summary_df = pd.DataFrame({
        "Metric": metric_names,
        "Original": orig_col,
        compName: comp_col
    })

    # Save CSV
    summary_df.to_csv(f"summary_table_{compName}.csv", index=False)

def wilcoxonTable(wilcoxonIterations, compName):
    # Save Wilcoxon Iterations
    W_stats = np.array([w[0] for w in wilcoxonIterations])
    p_vals  = np.array([w[1] for w in wilcoxonIterations])

    df = pd.DataFrame({
        "Iteration": np.arange(len(wilcoxonIterations)),
        "W_statistic": W_stats,
        "p_value": p_vals
    })

    df.to_csv(f"wilcoxon_table_{compName}.csv", index=False)

    # Save Averaged stats
    stats_df = pd.DataFrame({
        "Statistic": ["Mean p", "Std p", "Median p", "Min p", "Max p", "Significant (p<0.05)"],
        "Value": [
            p_vals.mean(),
            p_vals.std(),
            np.median(p_vals),
            p_vals.min(),
            p_vals.max(),
            np.sum(p_vals < 0.05)
        ]
    })
    stats_df.to_csv(f"wilcoxon_summary_stats_{compName}.csv", index=False)

    # Save a histogram
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6,4))
    sns.histplot(p_vals, bins=10, kde=False)
    plt.xlabel("p-value")
    plt.ylabel("Frequency")
    plt.title(f"Wilcoxon p-value distribution ({compName})")
    plt.tight_layout()
    plt.savefig(f"wilcoxon_p_hist_{compName}.png", dpi=300)
    plt.close()

def performanceMeasuresBarChart(averageResultsMatrix, compName):



    sns.set_theme(style="whitegrid")
    
    # Extract metrics
    n_iters = len(averageResultsMatrix)
    metric_names = [t[0] for t in averageResultsMatrix[0]]
    n_metrics = len(metric_names)

    orig_vals = np.zeros((n_iters, n_metrics))
    comp_vals = np.zeros((n_iters, n_metrics))

    for i in range(n_iters):
        for m in range(n_metrics):
            orig_vals[i, m] = averageResultsMatrix[i][m][1]
            comp_vals[i, m] = averageResultsMatrix[i][m][2]

    # Mean scores
    mean_orig = orig_vals.mean(axis=0)
    mean_comp = comp_vals.mean(axis=0)

    # Scale accuracy from % → [0–1]
    mean_orig[0] /= 100.0
    mean_comp[0] /= 100.0

    # Dataframe for plotting
    df_plot = pd.DataFrame({
        "Metric": metric_names * 2,
        "Model": ["Original"] * n_metrics + [compName] * n_metrics,
        "Value": list(mean_orig) + list(mean_comp)
    })

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_plot,
        x="Metric",
        y="Value",
        hue="Model",
        palette="deep"
    )

    plt.title(f"Performance Measures (Accuracy scaled 0–1)\nOriginal vs {compName}")
    plt.ylabel("Score")
    plt.xlabel("Metric")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"perf_measures_bar_{compName}.png", dpi=300)
    plt.close()

    print(f"Saved performance measure bar chart: perf_measures_bar_{compName}.png")

def plot_three_way_bar(final_stats):
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Convert the dictionary list into a Pandas DataFrame suitable for Seaborn
        # We need "Long Format": Metric | Model | Score
        plot_data = []
        
        for item in final_stats:
            metric = item["Metric"]
            # Add Original
            plot_data.append({"Metric": metric, "Model": "Original", "Score": item["Original"]})
            # Add RF
            plot_data.append({"Metric": metric, "Model": "Random Forest", "Score": item["Random Forest"]})
            # Add LR
            plot_data.append({"Metric": metric, "Model": "Logistic Regression", "Score": item["Logistic Regression"]})
            
        df = pd.DataFrame(plot_data)

        # Plotting
        sns.set_theme(style="ticks") # Matches the white background in your image
        plt.figure(figsize=(12, 6))
        
        chart = sns.barplot(
            data=df,
            x="Metric",
            y="Score",
            hue="Model",
            palette=["#1f77b4", "#ff7f0e", "#2ca02c"] # Classic matplotlib colors (Blue, Orange, Green)
        )
        
        plt.title("Performance Comparison: Original vs RF vs LR")
        plt.xticks(rotation=45)
        plt.legend(title=None) # Remove the "Model" title from legend to match image
        plt.tight_layout()
        
        filename = "three_way_comparison.png"
        plt.savefig(filename, dpi=300)
        print(f"Graph saved to {filename}")