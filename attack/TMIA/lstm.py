import torch
import torch.nn as nn

class BiLSTMChangePoint(nn.Module):

    def __init__(self, hidden_size: int = 32, fc_hidden_size: int = 32, use_stats: bool = False):

        super().__init__()
        self.use_stats = use_stats
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                            batch_first=True, bidirectional=True)

        if use_stats:
            input_dim = hidden_size * 2 + 4
        else:
            input_dim = hidden_size * 2

        self.fc1 = nn.Linear(input_dim, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 3)

    def forward(self, x: torch.Tensor, stats: torch.Tensor = None) -> torch.Tensor:
        out, _ = self.lstm(x)  # out shape: (batch_size, 10, hidden_size*2)

        seq_len = x.size(1)
        mid_idx = seq_len // 2
        mid_out = out[:, mid_idx, :]

        if self.use_stats:
            if stats is None:
                raise ValueError("use_stats=True but stats is None")
            combined = torch.cat([mid_out, stats], dim=1)  # (batch_size, hidden_size*2 + 4)
            features = combined
        else:
            features = mid_out

        x = self.fc1(features)
        x = self.relu(x)
        x = self.fc2(x)
        return x