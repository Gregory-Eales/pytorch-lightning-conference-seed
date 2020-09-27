import torch

from .encoder import Encoder
from .decoder import Decoder


class VAE(torch.nn.Module):
	"""
	variational auto encoder

	takes in raw image data and ouputs probabalistic decodings of the image
	"""
	def __init__(self, hparams):
		

		self.encoder = Encoder(hparams)
		self.decoder = Decoder(hparams)

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