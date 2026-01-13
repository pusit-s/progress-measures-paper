from fraction_config import FractionConfig, ExperimentConfig
from fraction_experiment import FractionExperiment

if __name__ == "__main__":
    # ===========================
    # User Configuration
    # ===========================
    
    # Define Fraction Configuration
    # You can adjust P, steps, learning rate etc. here
    frac_config = FractionConfig(
        p=113,
        total_steps=20000,
        log_every=200
    )

    # Define Experiment Configurations
    experiments = [
        # 1. Fraction range 0.1 to 0.9 (random shuffle)
        ExperimentConfig(
            name="experiment_1_fraction",
            style="random",
            values_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ),
        # 2. Square range k=30 to 60 with step 5 (rect)
        ExperimentConfig(
            name="experiment_2_square",
            style="rect",
            values_list=[30, 35, 40, 45, 50, 55, 60]
        ),
        # 3. Stripe range k=40 to 70 with step 10 (strip)
        ExperimentConfig(
            name="experiment_3_stripe",
            style="strip",
            values_list=[40, 50, 60, 70]
        )
    ]
    
    # ===========================
    # Execution
    # ===========================
    
    for exp_config in experiments:
        experiment = FractionExperiment(exp_config, frac_config)
        experiment.run()
