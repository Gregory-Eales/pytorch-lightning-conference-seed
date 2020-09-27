import torch
import pytorch_lightning as pl
from encoder import Encoder
from decoder import Decoder

class GAN(pl.LightningModule):

	def __init__(self, hparams):

		super(GAN, self).__init__()

		self.encoder = Encoder(hparams)


	def forward(self):
		pass


