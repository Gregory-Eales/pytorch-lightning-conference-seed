import torch



class Critic(torch.nn.Module):

	"""
	critic or value network used to make value predictions from states

	"""

	def __init__(self, hparams):

		self.l1 = torch.nn.Linear(100, 100)

		self.relu = torch.nn.LeakyReLU()


	def forward(self, x):

		pass