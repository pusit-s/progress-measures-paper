import torch
import torch.nn as nn
import torch.optim as optim
import sys
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
                
        return history
