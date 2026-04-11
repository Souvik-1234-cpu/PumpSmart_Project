# src/model_architecture.py
# Exact architecture reverse-engineered from lstm_ae_baseline_best.pth state_dict
# 
# Confirmed dimensions:
#   encoder.lstm1 : LSTM(8  → 128, num_layers=2)
#   encoder.lstm2 : LSTM(128 → 64,  num_layers=1)
#   encoder.bn    : LayerNorm(64)   ← has weight+bias but NO running_mean → LayerNorm, not BN
#   decoder.fc_h  : Linear(64 → 128)
#   decoder.fc_c  : Linear(64 → 128)
#   decoder.lstm1 : LSTM(64  → 128, num_layers=2)
#   decoder.lstm2 : LSTM(128 → 8,   num_layers=1)
#   decoder.out   : Linear(8 → 8)

import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(8,   128, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64,  num_layers=1, batch_first=True)
        self.bn    = nn.LayerNorm(64)   # named 'bn' to match checkpoint keys

    def forward(self, x):
        out1, _      = self.lstm1(x)
        out2, (h, c) = self.lstm2(out1)
        h_bn = self.bn(h[-1])           # shape: (batch, 64)
        return h_bn, h, c

class LSTMDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_h  = nn.Linear(64,  128)
        self.fc_c  = nn.Linear(64,  128)
        self.lstm1 = nn.LSTM(64,  128, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(128, 8,   num_layers=1, batch_first=True)
        self.out   = nn.Linear(8, 8)

    def forward(self, bottleneck, seq_len, h_enc, c_enc):
        # Project encoder bottleneck → decoder init state
        # .repeat(2,1,1) because decoder.lstm1 has num_layers=2
        h0 = torch.tanh(self.fc_h(h_enc[-1])).unsqueeze(0).repeat(2, 1, 1)
        c0 = torch.tanh(self.fc_c(c_enc[-1])).unsqueeze(0).repeat(2, 1, 1)
        x  = bottleneck.unsqueeze(1).repeat(1, seq_len, 1)
        out1, _ = self.lstm1(x, (h0, c0))
        out2, _ = self.lstm2(out1)
        return self.out(out2)

class LSTMAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = LSTMEncoder()
        self.decoder = LSTMDecoder()

    def forward(self, x):
        bottleneck, h, c = self.encoder(x)
        return self.decoder(bottleneck, x.size(1), h, c)