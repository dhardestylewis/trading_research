"""Sequence/Tabular Hybrid Challenger for exp026."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.utils.logging import get_logger

log = get_logger("sequence_challenger_exp026")

class SequenceTabularHybrid(nn.Module):
    """A simple hybrid PyTorch model taking both latest tabular state and sequence history."""
    def __init__(self, num_tabular_feat: int, num_seq_feat: int, seq_len: int, hidden_dim: int = 32):
        super().__init__()
        self.tabular_mlp = nn.Sequential(
            nn.Linear(num_tabular_feat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Sequence encoder (LSTM)
        self.seq_encoder = nn.LSTM(
            input_size=num_seq_feat,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x_tab, x_seq):
        tab_out = self.tabular_mlp(x_tab)
        _, (hn, _) = self.seq_encoder(x_seq)
        seq_out = hn[-1]
        combined = torch.cat([tab_out, seq_out], dim=1)
        return self.head(combined).squeeze(-1)

class ChallengerExp026:
    """Trainer and wrapper for SequenceTabularHybrid per horizon."""
    def __init__(self, horizons: list[int], seq_cols: list[str], seq_len: int = 6):
        self.horizons = horizons
        self.seq_cols = seq_cols
        self.seq_len = seq_len
        self.models = {}
        self.tabular_cols = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _prepare_tensors(self, X: pd.DataFrame, is_train: bool = False):
        # Extract sequence columns
        # They are named like seq_{col}_lag_{lag}
        # Shape: (batch, seq_len, num_features)
        
        if is_train:
            # Figure out tabular columns (everything not starting with seq_ or metadata)
            meta_cols = ["asset", "timestamp", "target"]
            # Exclude seq cols
            self.tabular_cols = [c for c in X.columns if not c.startswith("seq_") and c not in meta_cols]
            
        N = len(X)
        seq_tensor = np.zeros((N, self.seq_len, len(self.seq_cols)), dtype=np.float32)
        
        for f_idx, col in enumerate(self.seq_cols):
            for t_idx, lag in enumerate(range(1, self.seq_len + 1)):
                feat_name = f"seq_{col}_lag_{lag}"
                if feat_name in X.columns:
                    seq_tensor[:, self.seq_len - t_idx - 1, f_idx] = X[feat_name].fillna(0).values
                    
        tab_tensor = X[self.tabular_cols].fillna(0).values.astype(np.float32)
        
        return torch.FloatTensor(tab_tensor), torch.FloatTensor(seq_tensor)

    def fit(self, X_train: pd.DataFrame, horizons_df: pd.DataFrame):
        log.info(f"Fitting sequence challenger on device: {self.device}")
        
        for h in self.horizons:
            target_col = f"gross_move_bps_{h}"
            if target_col not in horizons_df.columns:
                continue
                
            y_train = horizons_df[target_col].astype(np.float32)
            valid_idx = ~y_train.isna()
            X_valid = X_train[valid_idx]
            y_valid = y_train[valid_idx]
            
            x_tab, x_seq = self._prepare_tensors(X_valid, is_train=True)
            y_t = torch.FloatTensor(y_valid.values)
            
            dataset = TensorDataset(x_tab, x_seq, y_t)
            loader = DataLoader(dataset, batch_size=256, shuffle=True)
            
            num_tab_feat = x_tab.shape[1]
            num_seq_feat = x_seq.shape[2]
            
            model = SequenceTabularHybrid(
                num_tabular_feat=num_tab_feat, 
                num_seq_feat=num_seq_feat, 
                seq_len=self.seq_len
            ).to(self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
            criterion = nn.MSELoss()
            
            model.train()
            epochs = 1
            for epoch in range(epochs):
                for b_tab, b_seq, b_y in loader:
                    b_tab = b_tab.to(self.device)
                    b_seq = b_seq.to(self.device)
                    b_y = b_y.to(self.device)
                    
                    optimizer.zero_grad()
                    preds = model(b_tab, b_seq)
                    loss = criterion(preds, b_y)
                    loss.backward()
                    optimizer.step()
                    
            model.eval()
            self.models[h] = model
            log.info(f"Fitted Sequence Challenger for {h} bars.")

    def predict(self, X_test: pd.DataFrame) -> dict[int, pd.DataFrame]:
        preds = {}
        x_tab, x_seq = self._prepare_tensors(X_test, is_train=False)
        x_tab = x_tab.to(self.device)
        x_seq = x_seq.to(self.device)
        
        for h, model in self.models.items():
            model.eval()
            with torch.no_grad():
                out = model(x_tab, x_seq).cpu().numpy()
                
            preds_df = pd.DataFrame(index=X_test.index)
            preds_df["pred_SequenceChallenger"] = out
            preds[h] = preds_df
            
        return preds
