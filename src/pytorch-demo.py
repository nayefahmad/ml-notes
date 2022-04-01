# # PyTorch Demo

# Reference:
# - J. Papa, PyTorch Pocket Reference, Chapter 1

import urllib.request

from torchvision import models

url = "https://pytorch.tips/coffee"
fpath = "data/coffee.jpg"
urllib.request.urlretrieve(url, fpath)


model = models.alexnet(pretrained=True)

print(model)
