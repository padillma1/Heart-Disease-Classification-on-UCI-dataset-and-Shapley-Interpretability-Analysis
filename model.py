#default log. reg.

import torch
import torch.nn as nn


#class Baseline(nn.Module):
#    def __init__(self, input_dim=13, output_dim=1):
#        super(Baseline, self).__init__()
#        self.linear = torch.nn.Linear(input_dim, output_dim)

#    def forward(self, x):
#        return torch.sigmoid(self.linear(x))

#one layer NN
#import torch
#import torch.nn as nn

#class Baseline(nn.Module):
#   def __init__(self, input_dim=13, hidden_dim=16, output_dim=1):
#        super(Baseline, self).__init__()
#        self.linear1 = nn.Linear(input_dim, hidden_dim)
#        self.relu = nn.ReLU()
#        self.linear2 = nn.Linear(hidden_dim, output_dim)

#   def forward(self, x):
#        x = self.linear1(x)
#        x = self.relu(x)
#        x = self.linear2(x)
#        return torch.sigmoid(x)


#for SVM
class Baseline(nn.Module):
    def __init__(self, input_dim=13, output_dim=1):
        super(Baseline, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


 