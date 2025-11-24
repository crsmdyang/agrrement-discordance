import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_exact_concordance_bar(summary_df: pd.DataFrame) -> plt.Figure:
    """
    Bar chart of agreement_rate (%) by pair_name.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    pairs = summary_df["pair_name"]
    rates = summary_df["agreement_rate"] * 100
    
    bars = ax.bar(pairs, rates, color=['#4c72b0', '#55a868', '#c44e52'])
    
    ax.set_ylabel('Agreement Rate (%)')
    ax.set_title('Exact Concordance Rates')
    ax.set_ylim(0, 100)
    
    # Add labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def plot_concordance_score_bar(descriptive_df: pd.DataFrame) -> plt.Figure:
    """
    Bar chart of mean_score ± SD by pair.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    pairs = descriptive_df["pair"]
    means = descriptive_df["mean_score"]
    stds = descriptive_df["sd_score"]
    
    ax.bar(pairs, means, yerr=stds, capsize=5, color=['#8172b3', '#ccb974', '#64b5cd'], alpha=0.8)
    
    ax.set_ylabel('Mean Concordance Score (0-5)')
    ax.set_title('Concordance Ratings by Experts (Mean ± SD)')
    ax.set_ylim(0, 5.5)
    
    plt.tight_layout()
    return fig

def plot_discordance_reasons(df_reasons: pd.DataFrame) -> plt.Figure:
    """
    Bar chart of n by discordance_reason_category.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by count
    df_sorted = df_reasons.sort_values("n", ascending=False)
    
    ax.bar(df_sorted["discordance_reason_category"], df_sorted["n"], color='salmon')
    
    ax.set_ylabel('Count')
    ax.set_xlabel('Reason Category')
    ax.set_title('Frequency of Discordance Reasons')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig
