import matplotlib.pyplot as plt
import numpy as np
from fraction_config import FractionConfig
from fraction_data import FractionDataManager

class FractionVisualizer:
    def __init__(self, config: FractionConfig):
        self.config = config
        self.data_manager = FractionDataManager(config.p, config.device)

    def plot_dataset_distribution(self, ax, train_x, test_x, p):
        """
        Visualizes the dataset distribution (Train vs Test).
        Train_x and Test_x are (N, 2) tensors containing (i, j) pairs.
        """
        # Convert to CPU numpy for plotting
        def to_cpu(t):
            return t.cpu().numpy()
            
        train_data = to_cpu(train_x)
        test_data = to_cpu(test_x)
        
        # Plot Test first (background)
        if len(test_data) > 0:
            ax.scatter(test_data[:, 1], test_data[:, 0], c='lightgrey', s=2, marker='s', label='Test')
            
        # Plot Train
        if len(train_data) > 0:
            ax.scatter(train_data[:, 1], train_data[:, 0], c='tab:blue', s=2, marker='s', label='Train')
            
        ax.set_xlim(-0.5, p - 0.5)
        ax.set_ylim(-0.5, p - 0.5)
        ax.invert_yaxis() # Put row 0 at the top
        ax.set_aspect('equal')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    def generate_plots(self, df, name, style, values_list):
        """
        Generates a combined figure for the experiment batch.
        Rows: Each value in values_list
        Cols: 3 (Distribution, Loss, Accuracy)
        """
        num_vals = len(values_list)
        cols = 3
        rows = num_vals
        
        # Increase figure height to accommodate all rows
        fig_width = 15 # 5 * 3
        fig_height = 4 * rows
        
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        
        # Handle single row case where axes is 1D array
        if rows == 1:
            axes = np.array([axes])
            
        for i, val in enumerate(values_list):
            # --- Column 1: Dataset Distribution ---
            ax_dist = axes[i, 0]
            
            if (style == 'rect' or style == 'strip'):
                frac = 0.3
                k_val = val
            else:
                frac = val
                k_val = 0
            
            # Using DataManager to regenerate the data for plotting visualization accurately
            train_x, _, test_x, _ = self.data_manager.create_data(
                fraction=frac, style=style, k=k_val, seed=self.config.seed
            )
            
            self.plot_dataset_distribution(ax_dist, train_x, test_x, self.config.p)
            
            label_title = f"{style.capitalize()} " + (f"Fraction {val}" if style == 'random' else f"k = {val}")
            ax_dist.set_title(label_title, fontsize=10, fontweight='bold')
            if i == 0: 
                ax_dist.legend(loc='upper right', fontsize=8)

            # Filter data for this value
            subset = df[df['variable'] == val]
            
            # --- Column 2: Loss ---
            ax_loss = axes[i, 1]
            
            if not subset.empty:
                ax_loss.semilogy(subset['step'], subset['train_loss'], label='Train', color='tab:blue')
                ax_loss.semilogy(subset['step'], subset['test_loss'], label='Test', color='lightgrey') 
                ax_loss.set_title("Loss", fontsize=10)
                ax_loss.grid(True, which="both", ls="--", alpha=0.3)
                if i == 0: ax_loss.legend()
            else:
                ax_loss.text(0.5, 0.5, "No Data", ha='center', va='center')

            # --- Column 3: Accuracy ---
            ax_acc = axes[i, 2]
            
            if not subset.empty:
                ax_acc.plot(subset['step'], subset['train_acc'], label='Train', color='tab:blue')
                ax_acc.plot(subset['step'], subset['test_acc'], label='Test', color='grey') 
                ax_acc.set_title("Accuracy", fontsize=10)
                ax_acc.set_ylim(-0.05, 1.05)
                ax_acc.grid(True, alpha=0.3)
                if i == 0: ax_acc.legend(loc='lower right')
            else:
                ax_acc.text(0.5, 0.5, "No Data", ha='center', va='center')

        plt.tight_layout()
        plot_filename = f"{name}_combined.png"
        plt.savefig(plot_filename, dpi=150)
        print(f"Saved combined plots to {plot_filename}")
