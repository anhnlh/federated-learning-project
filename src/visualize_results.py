import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_metric(metric, title, ylabel, folder):
    plt.figure(figsize=(10, 6))
    
    client_dfs = []
    for file in os.listdir(folder):
        if file.startswith("client_") and file.endswith("_metrics.csv"):
            df = pd.read_csv(os.path.join(folder, file))
            plt.plot(df["round"], df[metric], alpha=0.5, linestyle='--')
            client_dfs.append(df)
    
 
    avg_df = pd.concat(client_dfs).groupby("round").mean().reset_index()
    plt.plot(avg_df["round"], avg_df[metric], label="Average", linewidth=2, color="red")
    
    global_df = pd.read_csv(os.path.join(folder, "global_metrics.csv"))
    plt.plot(global_df["round"], global_df[metric], label="Global", linewidth=2, color="black")
    
    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(folder, f"{metric}_history.png"))
    plt.close()

def generate_plots(folder):
    plot_metric("loss", f"Loss History - {folder.split('/')[-1]}", "Loss", folder)
    plot_metric("accuracy", f"Accuracy History - {folder.split('/')[-1]}", "Accuracy", folder)

if __name__ == "__main__":
    generate_plots("results/no_attack")
    generate_plots("results/attack")
    print("Plots generated successfully.")