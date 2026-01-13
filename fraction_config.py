from dataclasses import dataclass, field
from typing import List, Optional
import torch

@dataclass
class FractionConfig:
    p: int = 113
    total_steps: int = 20000
    log_every: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1.0
    seed: int = 42
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Activation Collection
    collect_activations: bool = True
    collect_every: int = 100
    collect_layers: List[str] = field(default_factory=lambda: ['blocks.0.mlp.hook_post'])
    save_dir: str = "activations"

    # Transformer Config
    d_model: int = 128
    num_heads: int = 4
    d_mlp: int = 512
    num_layers: int = 1
    n_ctx: int = 2
    act_type: str = 'ReLU'
    use_ln: bool = False

@dataclass
class ExperimentConfig:
    name: str
    style: str  # 'random', 'strip', 'rect'
    values_list: List[float] # or List[int] depending on style, but float handles both roughly
