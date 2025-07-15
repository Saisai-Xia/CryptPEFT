import torch
import torch.nn as nn
import torch.nn.functional as F

class Controller(nn.Module):
    def __init__(self, num_choices_list, hidden_size=64, mlp_depth=2):
        super().__init__()
        self.num_choices_list = num_choices_list
        self.hidden_size = hidden_size
        self.total_choices = sum(num_choices_list)

        
        self.init_embedding = nn.Parameter(torch.zeros(hidden_size))

        
        mlp_layers = []
        in_dim = hidden_size
        for _ in range(mlp_depth - 1):
            mlp_layers.append(nn.Linear(in_dim, hidden_size))
            mlp_layers.append(nn.ReLU())
            in_dim = hidden_size
        mlp_layers.append(nn.Linear(in_dim, self.total_choices))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self):
        actions = []
        log_probs = []

        
        hidden = self.init_embedding
        logits_all = self.mlp(hidden)  
        
        split_logits = torch.split(logits_all, self.num_choices_list)

        for i, logits in enumerate(split_logits):
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            actions.append(action)
            log_probs.append(log_prob)

        return actions, log_probs
