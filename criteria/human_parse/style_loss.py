
import torch
from torch import nn
import torchvision.models as models
class VGG19(torch.nn.Module):
  def __init__(self):
    super(VGG19, self).__init__()
    features = models.vgg19(pretrained=True).features
    self.relu1_1 = torch.nn.Sequential()
    self.relu1_2 = torch.nn.Sequential()

    self.relu2_1 = torch.nn.Sequential()
    self.relu2_2 = torch.nn.Sequential()

    self.relu3_1 = torch.nn.Sequential()
    self.relu3_2 = torch.nn.Sequential()
    self.relu3_3 = torch.nn.Sequential()
    self.relu3_4 = torch.nn.Sequential()

    self.relu4_1 = torch.nn.Sequential()
    self.relu4_2 = torch.nn.Sequential()
    self.relu4_3 = torch.nn.Sequential()
    self.relu4_4 = torch.nn.Sequential()

    self.relu5_1 = torch.nn.Sequential()
    self.relu5_2 = torch.nn.Sequential()
    self.relu5_3 = torch.nn.Sequential()
    self.relu5_4 = torch.nn.Sequential()

    for x in range(2):
      self.relu1_1.add_module(str(x), features[x])

    for x in range(2, 4):
      self.relu1_2.add_module(str(x), features[x])

    for x in range(4, 7):
      self.relu2_1.add_module(str(x), features[x])

    for x in range(7, 9):
      self.relu2_2.add_module(str(x), features[x])

    for x in range(9, 12):
      self.relu3_1.add_module(str(x), features[x])

    for x in range(12, 14):
      self.relu3_2.add_module(str(x), features[x])

    for x in range(14, 16):
      self.relu3_2.add_module(str(x), features[x])

    for x in range(16, 18):
      self.relu3_4.add_module(str(x), features[x])

    for x in range(18, 21):
      self.relu4_1.add_module(str(x), features[x])

    for x in range(21, 23):
      self.relu4_2.add_module(str(x), features[x])

    for x in range(23, 25):
      self.relu4_3.add_module(str(x), features[x])

    for x in range(25, 27):
      self.relu4_4.add_module(str(x), features[x])

    for x in range(27, 30):
      self.relu5_1.add_module(str(x), features[x])

    for x in range(30, 32):
      self.relu5_2.add_module(str(x), features[x])

    for x in range(32, 34):
      self.relu5_3.add_module(str(x), features[x])

    for x in range(34, 36):
      self.relu5_4.add_module(str(x), features[x])

    # don't need the gradients, just want the features
    for param in self.parameters():
      param.requires_grad = False

  def forward(self, x):
    relu1_1 = self.relu1_1(x)
    relu1_2 = self.relu1_2(relu1_1)

    relu2_1 = self.relu2_1(relu1_2)
    relu2_2 = self.relu2_2(relu2_1)

    relu3_1 = self.relu3_1(relu2_2)
    relu3_2 = self.relu3_2(relu3_1)
    relu3_3 = self.relu3_3(relu3_2)
    relu3_4 = self.relu3_4(relu3_3)

    relu4_1 = self.relu4_1(relu3_4)
    relu4_2 = self.relu4_2(relu4_1)
    relu4_3 = self.relu4_3(relu4_2)
    relu4_4 = self.relu4_4(relu4_3)

    relu5_1 = self.relu5_1(relu4_4)
    relu5_2 = self.relu5_2(relu5_1)
    relu5_3 = self.relu5_3(relu5_2)
    relu5_4 = self.relu5_4(relu5_3)

    out = {
      'relu1_1': relu1_1,
      'relu1_2': relu1_2,

      'relu2_1': relu2_1,
      'relu2_2': relu2_2,

      'relu3_1': relu3_1,
      'relu3_2': relu3_2,
      'relu3_3': relu3_3,
      'relu3_4': relu3_4,

      'relu4_1': relu4_1,
      'relu4_2': relu4_2,
      'relu4_3': relu4_3,
      'relu4_4': relu4_4,

      'relu5_1': relu5_1,
      'relu5_2': relu5_2,
      'relu5_3': relu5_3,
      'relu5_4': relu5_4,
    }
    return out


class StyleLoss(nn.Module):
  r"""
  Perceptual loss, VGG-based
  https://arxiv.org/abs/1603.08155
  https://github.com/dxyang/StyleTransfer/blob/master/utils.py
  """

  def __init__(self):
    super(StyleLoss, self).__init__()
    self.add_module('vgg_1', VGG19())
    self.criterion = torch.nn.L1Loss()

  def compute_gram(self, x):
    b, ch, h, w = x.size()
    f = x.view(b, ch, w * h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (h * w * ch)

    return G

  def __call__(self, x, y):

    # Compute features
    x_vgg, y_vgg = self.vgg_1(x), self.vgg_1(y)

    # Compute loss
    style_loss = 0.0
    style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
    style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
    style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
    style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

    return style_loss