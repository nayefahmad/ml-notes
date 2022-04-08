# # PyTorch Demo

# ## Overview

# This demo shows how to use a pretrained PyTorch model to predict the class
# to which an image belongs.


# Reference:
# - J. Papa, *PyTorch Pocket Reference*, Chapter 1

# ## Libraries

import urllib.request
import torch
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
            mean=[0.485, 0.456, 0.406],
            std=[
                0.229,
                0.224,
                0.225,
            ],  # precomputed based on the data used to train the model  # noqa
        ),
    ]
)

img_tensor = transform(img)

img_tensor.dim()  # Rank 3 tensor
img_tensor.shape  # 3 color channels, 224 * 224 pixels
img_tensor.size()  # 3 color channels, 224 * 224 pixels


# Showing tensor subsetting operations:

img_tensor[0, 0, 0]
img_tensor[:, 0, 0].shape
img_tensor[0, :, 0].shape
img_tensor[0, 0, :].shape


# Showing the three colour channels:

fig = plt.figure()
for channel_num, cmap in enumerate(["Reds", "Greens", "Blues"]):
    ax = plt.subplot(3, 1, channel_num + 1)
    plt.imshow(img_tensor[channel_num, :, :], cmap=cmap)
fig.show()


# ## Batching

# Efficient ML processes data in batches, and our model will expect a
# batch of data. Here we create a batch of size 1.

batch = img_tensor.unsqueeze(0)  # unsqueeze() adds a dimension to the tensor  # noqa
batch.dim()  # Rank 4 tensor
batch.shape


# ## Model

model = models.alexnet(pretrained=True)
print(model)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Set the module in evaluation mode. This has any effect only on
# certain modules.

model.eval()
model.to(device)
y = model(batch.to(device))
print(y.shape)

# First dimension has size 1 because there's only one image.
# Second dimension has size 1000 because there are 1000 classes.

# Finding winning class:

y_max, index = torch.max(y, 1)
print(index, y_max)

# What does class with index 967 represent? Let's find out.
