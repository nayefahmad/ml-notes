# # PyTorch Demo

# ## Overview

# This demo shows how to use a pretrained PyTorch model to predict the class
# to which an image belongs.


# Reference:
# - J. Papa, *PyTorch Pocket Reference*, Chapter 1

# ## Libraries

import urllib.request
from torchvision import models, transforms
import matplotlib.pyplot as plt
from PIL import Image

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

# ## Image to classify

url = "https://pytorch.tips/coffee"
fpath = "data/coffee.jpg"

re_download_image = False
if re_download_image:
    urllib.request.urlretrieve(url, fpath)

img = Image.open(fpath)
plt.imshow(img)
plt.show()

print(f"Image dimensions (width x height): {img.width} x {img.height}")

# ## Image transformation pipeline

out_width_and_height = 224

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(out_width_and_height),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # noqa
    ]
)

img_tensor = transform(img)

img_tensor.dim()  # Rank 3 tensor
img_tensor.size()  # 3 color channels, 224 * 224 pixels

img_tensor[0, 0, 0]
img_tensor[:, 0, 0].size()
img_tensor[0, :, 0].size()
img_tensor[0, 0, :].size()


# todo: how to show RGB?
# todo
for channel in range(3):
    plt.imshow(img_tensor[channel, :, :])
    plt.show()

# ## Model

model = models.alexnet(pretrained=True)

print(model)
