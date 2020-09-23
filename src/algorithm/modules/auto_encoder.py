import torch

from .encoder import Encoder
from .decoder import Decoder

class AutoEncoder(torch.nn.Module):

	"""
	this is an auto encoder used to learn latent representations of the data
	"""

	def __init__(self, hparams):
		
		self.encoder = Encoder()
		self.decoder = Decoder()

		self.l_mu = nn.Linear(1024, 1024)
        self.l_logvar = nn.Linear(1024, 1024)

	def forward(self, x):

		out = self.encoder(x)

		logvar = self.l_logvar(out)
        mu = self.l_mu(out)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return mu + eps*std

    def loss(self, x, encoding):

    	prediction = self.decoder(encoding)

    	l1 = torch.nn.NLLLoss(prediction, x)
    	l2 = torch.nn.KLDivLoss(prediction, x)

    	return l1 + l2
    	


