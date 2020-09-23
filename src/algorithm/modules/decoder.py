from torch import nn
import torch

class Decoder(nn.Module):

    def __init__(self, hparams):
        super(Decoder, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2)

    def forward(self, x):

        out = x.reshape(-1, 256*4, 1, 1)

        out = self.deconv1(out)
        out = nn.functional.relu(out)
        out = self.deconv2(out)
        out = nn.functional.relu(out)
        out = self.deconv3(out)
        out = nn.functional.relu(out)
        out = self.deconv4(out)
        out = torch.sigmoid(out)

        return out

if __name__ == "__main__":


    x = torch.ones(10, 3, 64, 64)

    print(x.shape)

    encoder = LargeEncoder()
    decoder = LargeDecoder()
  

    y = encoder.forward(x)

    print(y.shape)

    y = decoder.forward(y)

    print(y.shape)