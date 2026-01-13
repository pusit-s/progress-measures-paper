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

    # Define Experiment Configuration
    # style can be: 'random', 'strip', 'rect'
    # values_list depends on style:
    # - random: list of fractions (0.0 - 1.0)
    # - strip/rect: list of k values (integers)
    
    # Example 1: Random Fraction
    # exp_config = ExperimentConfig(
    #     name="experiment_random",
    #     style="random",
    #     values_list=[0.1, 0.3, 0.5, 0.7, 0.9]
    # )

    # Example 2: Square Holds (Rect)
    exp_config = ExperimentConfig(
        name="experiment_rect_demo-test",
        style="rect",
        values_list=[30, 45, 60]
    )

    # Example 3: Strip Holds
    # exp_config = ExperimentConfig(
    #     name="experiment_strip",
    #     style="strip",
    #     values_list=[40, 50, 60]
    # )
    
    # ===========================
    # Execution
    # ===========================
    
    experiment = FractionExperiment(exp_config, frac_config)
    experiment.run()
