import pandas as pd
from tqdm.auto import tqdm
from fraction_config import FractionConfig, ExperimentConfig
from fraction_trainer import FractionTrainer
from fraction_visualizer import FractionVisualizer

class FractionExperiment:
    def __init__(self, exp_config: ExperimentConfig, frac_config: FractionConfig):
        self.exp_config = exp_config
        self.frac_config = frac_config
        self.trainer = FractionTrainer(frac_config)
        self.visualizer = FractionVisualizer(frac_config)

    def run(self):
        print(f"\n=== Starting Experiment: {self.exp_config.name} (Style: {self.exp_config.style}) ===")
        print(f"Values to run: {self.exp_config.values_list}")
        
        all_data = []
        
        # Run training for each value in the list
        for val in tqdm(self.exp_config.values_list, desc=f"Running {self.exp_config.name}"):
            run_history = self.trainer.train_single_run(val, self.exp_config.style)
            all_data.extend(run_history)
            
        # Save CSV
        df = pd.DataFrame(all_data)
        csv_filename = f"{self.exp_config.name}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Saved data to {csv_filename}")
        
        # Generate Plots
        print("Generating plots...")
        self.visualizer.generate_plots(df, self.exp_config.name, self.exp_config.style, self.exp_config.values_list)
