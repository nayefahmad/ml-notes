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
from pathlib import Path

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

proj_path_string = r"C:/Nayef/ml-notes"  # os.path.dirname(__file__) doesn't work in Python Console or Jupyter # noqa
proj_path = Path(proj_path_string)

# ## Image to classify

url = "https://pytorch.tips/coffee"
fpath = proj_path.joinpath("data/coffee.jpg").as_posix()

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

# Applying the pipeline:

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
plot_rows = 3
plot_cols = 1
for channel_num, cmap in enumerate(["Reds", "Greens", "Blues"]):
    ax = plt.subplot(plot_rows, plot_cols, channel_num + 1)
    plt.imshow(img_tensor[channel_num, :, :], cmap=cmap)
fig.show()


# ## Batching

# Efficient ML processes data in batches, and our model will expect a
# batch of data. Here we create a batch of size 1.

batch = img_tensor.unsqueeze(0)  # unsqueeze() adds a dimension to the tensor  # noqa
batch.dim()  # Rank 4 tensor
batch.shape


# ## Model setup

model = models.alexnet(pretrained=True)
print(model)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Set the model in evaluation mode. This has an effect only on
# certain components of the model.

model.eval()
model.to(device)

# ## Passing data to the model

y = model(batch.to(device))
print(y.shape)

# First dimension has size 1 because there's only one image.
# Second dimension has size 1000 because there are 1000 classes.

# Finding most-likely class:

y_max, index = torch.max(y, 1)
print(index, y_max)

# What does class with index 967 represent? Let's find out.

url = "https://pytorch.tips/imagenet-labels"
fpath = proj_path.joinpath("data/imagenet_class_labels.txt").as_posix()

re_download_file = False
if re_download_file:
    urllib.request.urlretrieve(url, fpath)

with open(fpath) as f:
    classes = [line.strip() for line in f.readlines()]

print(classes[index.item()])

# Extracting probabilities

probs_nonflat = torch.nn.functional.softmax(y, dim=1)
probs = probs_nonflat[0]

print(f"Most likely class is {classes[index.item()]}")
print(
    f"Probability of this class is "
    f"{round(probs[index.item()].item() * 100, 2)}"  # noqa
)


# ## Extracting top 5 most likely classes

_, indices_nonflat = torch.sort(y, descending=True)
indices = indices_nonflat[0]

for i in indices[:5]:
    print(f"class: {classes[i]} ; prob: {round(probs[i].item() *100, 2)}")


# ## Trying out other images
