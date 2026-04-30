import math
import torch
import torch.nn as nn

class Temperature(nn.Module):
    def __init__(self, init_temperature: float = 1.0):
        super().__init__()
        init_temperature = max(float(init_temperature), 1e-6)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(init_temperature), dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return self.log_temperature.exp()
