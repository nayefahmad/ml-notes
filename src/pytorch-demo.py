# # PyTorch Demo

# ## Overview

# This demo shows how to use a pretrained PyTorch model to predict the class
# to which an image belongs.


# Reference:
# - J. Papa, *PyTorch Pocket Reference*, Chapter 1

import urllib.request
from torchvision import models

url = "https://pytorch.tips/coffee"
fpath = "data/coffee.jpg"
urllib.request.urlretrieve(url, fpath)


model = models.alexnet(pretrained=True)

print(model)
