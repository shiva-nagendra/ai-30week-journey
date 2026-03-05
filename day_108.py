#week 18 day 1
#Convolutional Neural Networks (CNN)

import torch
import torch.nn as nn

#Create a fake image(1 channel, 28x28 MNIST)
image = torch.randn(1, 1, 28,28)

print("Input image shape:")
print(image.shape)

#Define convolution layer
conv = nn.Conv2d(
    in_channels=1,
    out_channels=8,
    kernel_size=3
)

#Apply convolution
feature_map = conv(image)

print("\nAfter convolution:")
print(feature_map.shape)

#Apply activation
relu = nn.ReLU()
activated = relu(feature_map)

print("\nAfter reLu:")
print(activated.shape)


#Apply pooling
pool = nn.MaxPool2d(kernel_size=2)

pooled = pool(activated)

print("\nAfter pooling:")
print(pooled.shape)
