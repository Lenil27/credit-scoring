import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_bars(df: pd.DataFrame,
              feature_col: str,
              target_col: str,
              ax: plt.Axes,
              width: float = 0.35,
              palette: tuple[str, str] = ('lightblue', 'lightcoral')
             ) -> plt.Axes:
    
    categories = sorted(df[feature_col].unique())
    
    for i, types in enumerate(categories):
        data = df[df[feature_col] == types]
        target_count = data[target_col].value_counts()
        target_1_val = target_count.get(1, 0)
        target_0_val = target_count.get(0, 0)
    
        bars_target_1 = ax.bar(i + width/2, target_1_val, width, color=palette[1])
        bars_target_0 = ax.bar(i - width/2, target_0_val, width, color=palette[0])
    
        all_bars = list(bars_target_0) + list(bars_target_1)
        all_val = [target_0_val, target_1_val]
    
        for bar, val in zip(all_bars, all_val):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 2,
                        str(int(val)),
                        ha='center', va='bottom', fontweight='bold', fontsize=9
                       )
            
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories, rotation= 15)
    ax.set_xlabel(feature_col.replace('_', ' ').title())
    ax.set_ylabel('Количество')

    return ax

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_grid_boxplot(df: pd.DataFrame,
                      num_features: list,
                      target: str,
                      palette: tuple[str, str] = ('lightblue', 'lightcoral')
                     ) -> None:
    n = len(num_features)
    rows = (n + 2) // 3
    
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4.5 * rows), squeeze=False)

    for i, col in enumerate(num_features):
        ax = axes[i // 3, i % 3]
        sns.boxplot(data=df, x=target, y=col, ax=ax, palette=palette, hue=target)

    for j in range(n, rows * 3):
        axes[j // 3, j % 3].set_visible(False)
        
    plt.tight_layout()
    plt.show()

def plot_grid_hist(df: pd.DataFrame,
                   num_features: list[str],
                   target: str,
                   palette: tuple[str, str] = ('lightblue', 'lightcoral')
                  ) -> None:
    
    n = len(num_features)
    rows = (n + 2) // 3

    fig, axes = plt.subplots(rows, 3, figsize=(15, 4.5 * rows), squeeze=False)

    for i, col in enumerate(num_features):
        ax = axes[i // 3, i % 3]
        sns.histplot(
            data=df,
            x=col,
            hue=target,
            stat='density',
            common_norm=False,
            kde=True,
            element='step', 
            palette=palette,
            ax=ax,
        )

    for j in range(n, rows * 3):
        axes[j // 3, j % 3].set_visible(False)

    plt.tight_layout()
    plt.show()