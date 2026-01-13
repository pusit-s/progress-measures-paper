import torch
import random
import sys

# Append current directory to path to find transformers if needed, though likely not needed if in same dir
sys.path.append('.')

class FractionDataManager:
    def __init__(self, p: int, device: torch.device):
        self.p = p
        self.device = device

    def to_tensors(self, pairs_list):
        if not pairs_list:
            return torch.empty((0, 2), dtype=torch.long).to(self.device), torch.empty((0), dtype=torch.long).to(self.device)
        
        data = torch.tensor(pairs_list, dtype=torch.long)
        x = data[:, 0]
        y = data[:, 1]
        labels = (x + y) % self.p
        inputs = torch.stack([x, y], dim=1)
        return inputs.to(self.device), labels.to(self.device)

    def create_data(self, fraction=0.3, style=None, k=30, seed=42):
        """
        Generates data based on the specific style:
        - random (or None): Uses 'fraction' to split train/test.
        - strip: Holds out rows 0..k for Test. Samples 'fraction' of remaining for Train.
        - rect: Holds out square 0..k x 0..k for Test. Samples 'fraction' of remaining for Train.
        """
        random.seed(seed)
        all_pairs = [(i, j) for i in range(self.p) for j in range(self.p)]
        
        test_set = set()
        if style == 'strip':
            test_set = {(i, j) for i in range(k + 1) for j in range(self.p)}
        elif style == 'rect':
            test_set = {(i, j) for i in range(k + 1) for j in range(k + 1)}
        
        remaining = [p for p in all_pairs if p not in test_set]
        random.shuffle(remaining)
        num_train = int(fraction * len(remaining))
        train_pairs = remaining[:num_train]
        test_pairs = [p for p in all_pairs if p in test_set] if test_set else remaining[num_train:]

        train_x, train_y = self.to_tensors(train_pairs)
        test_x, test_y = self.to_tensors(test_pairs)
        
        return train_x, train_y, test_x, test_y
