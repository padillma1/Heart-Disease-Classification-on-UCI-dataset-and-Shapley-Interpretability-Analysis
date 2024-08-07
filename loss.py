import torch
from torch.nn.modules.loss import _Loss
import torch.nn as nn


#class BaselineLoss(_Loss):
#    def __init__(self):
#       super(BaselineLoss, self).__init__()
#        self.bce = torch.nn.BCELoss()

#    def forward(self, input: torch.Tensor, target: torch.Tensor):
#        return self.bce(input, target)

#hinge loss for SVM model

class BaselineLoss(_Loss):
	def __init__(self):
		super(BaselineLoss,self).__init__()
		self.hinge = nn.HingeEmbeddingLoss()

	def forward(self,input: torch.Tensor, target: torch.Tensor):
		target = 2 * target - 1 # convert labels from {0, 1} to {-1, 1}
		hinge_loss = 1 - torch.mul(input,target)
		hinge_loss[hinge_loss < 0] = 0 # equivalent to max(0, 1 - y*y_pred)
		return hinge_loss.mean()
        

