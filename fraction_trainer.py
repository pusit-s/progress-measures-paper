import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim as optim
import sys
import os
from fraction_config import FractionConfig
from fraction_data import FractionDataManager

# Import transformer
sys.path.append('.')
from transformers import Transformer, Config as TransConfig

class FractionTrainer:
    def __init__(self, config: FractionConfig):
        self.config = config
        self.data_manager = FractionDataManager(config.p, config.device)

    def evaluate(self, model, x, y, criterion):
        if len(x) == 0: return 0.0, 0.0
        model.eval()
        with torch.no_grad():
            logits = model(x)
            final_logits = logits[:, -1, :]
            loss = criterion(final_logits, y)
            preds = final_logits.argmax(dim=-1)
            acc = (preds == y).float().mean().item()
        model.train()
        return loss.item(), acc

    def train_single_run(self, val, style):
        """
        Trains a single model.
        val: is 'fraction' if style='random', else it is 'k'.
        """
        print(f"Training run: style={style}, val={val}")
        
        if (style == 'rect' or style == 'strip'):
            frac = 0.3
            k_val = val
        else:
            frac = val
            k_val = 0 
        
        # Generate Data
        train_x, train_y, test_x, test_y = self.data_manager.create_data(
            fraction=frac, style=style, k=k_val, seed=self.config.seed
        )
        
        # Model Config
        cfg = TransConfig(
            p=self.config.p,
            d_model=self.config.d_model,
            num_heads=self.config.num_heads,
            d_mlp=self.config.d_mlp,       
            num_layers=self.config.num_layers,
            n_ctx=self.config.n_ctx,
            d_vocab=self.config.p+1,
            act_type=self.config.act_type,
            use_ln=self.config.use_ln,
            seed=self.config.seed
        )
        
        model = Transformer(cfg).to(self.config.device)
        optimizer = optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        history = []
        
        # Activation Collection setup
        activations_history = {}
        all_data = None
        if self.config.collect_activations:
            # Generate all possible data pairs (0..p-1, 0..p-1)
            # Create a simple list of pairs.
            all_pairs = [(i, j) for i in range(self.config.p) for j in range(self.config.p)]
            # Use data manager to get tensors
            all_data, _ = self.data_manager.to_tensors(all_pairs)

        # Training Loop
        for step in range(self.config.total_steps + 1):
            model.train()
            optimizer.zero_grad()
            
            if len(train_x) > 0:
                logits = model(train_x)
                loss = criterion(logits[:, -1, :], train_y)
                loss.backward()
                optimizer.step()
            else:
                loss = torch.tensor(0.0)
            
            if step % self.config.log_every == 0:
                train_loss, train_acc = self.evaluate(model, train_x, train_y, criterion)
                test_loss, test_acc = self.evaluate(model, test_x, test_y, criterion)
                
                history.append({
                    'variable': val,
                    'step': step,
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'train_acc': train_acc,
                    'test_acc': test_acc
                })
            
            # Collect Activations
            if self.config.collect_activations and (step % self.config.collect_every == 0):
                model.eval()
                cache = {}
                with torch.no_grad():
                    # We need to use hook mechanism from transformers.py
                    # The Transformer class has cache_all method
                    model.remove_all_hooks()
                    model.cache_all(cache)
                    
                    # Forward pass on all_data
                    # all_data shape is [batch, 2]
                    # Data needs to be formatted as (x, y, p) tuples for config.fn or just passed to model? 
                    # The model expects [batch, 2] tensor for embedding if simplified, 
                    # BUT looking at transformers.py:
                    # forward takes x. x is [batch, 2]? 
                    # Embed.forward uses x.
                    # Let's check Embed.forward: returns t.einsum('dbp -> bpd', self.W_E[:, x])
                    # x should be indices. so [batch, n_ctx]
                    
                    # Our all_data is [batch, 2] (x, y)
                    
                    # We need access to intermediate activations.
                    # We can use the cache populated by cache_all.
                    
                    _ = model(all_data)
                    
                    step_acts = {}
                    for layer_name in self.config.collect_layers:
                        if layer_name in cache:
                            # Move to CPU to save memory
                            step_acts[layer_name] = cache[layer_name].cpu()
                    
                    activations_history[step] = step_acts
                    
                    model.remove_all_hooks()
                model.train()

        # Save activations
        if self.config.collect_activations:
            os.makedirs(self.config.save_dir, exist_ok=True)
            save_path = os.path.join(self.config.save_dir, f"{style}_{val}_activations.pth")
            torch.save(activations_history, save_path)
            print(f"Saved activations to {save_path}")
                
        return history
