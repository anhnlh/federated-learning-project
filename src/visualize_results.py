import glob

import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_metric(metric, title, ylabel, folder):
    plt.figure(figsize=(10, 6))
    
    client_dfs = []
    for file in os.listdir(folder):
        if file.startswith("client_") and file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, file))
            plt.plot(df["round"], df[metric], alpha=0.5, linestyle='--')
            client_dfs.append(df)
    
 
    avg_df = pd.concat(client_dfs).groupby("round").mean().reset_index()
    plt.plot(avg_df["round"], avg_df[metric], label="Average", linewidth=2, color="red")
    
    global_file = glob.glob(os.path.join(folder, "global_metrics(*).csv"))[0]
    global_df = pd.read_csv(global_file)
    plt.plot(global_df["round"], global_df[metric], label="Global", linewidth=2, color="black")
    
    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(folder, f"{metric}_history.png"))
    plt.close()

def generate_plots(folder, is_poisoned):
    plot_metric("loss", f"Loss History - {'poisoned' if is_poisoned else 'unpoisoned'}", "Loss", folder)
    plot_metric("accuracy", f"Accuracy History - {'poisoned' if is_poisoned else 'unpoisoned'}", "Accuracy", folder)

if __name__ == "__main__":
    generate_plots("results/no_attack", False)
    generate_plots("results/attack", True)
    print("Plots generated successfully.")