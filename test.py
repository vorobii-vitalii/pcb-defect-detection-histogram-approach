import PIL
import torch
from torch._C._te import Tensor
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

NEW_MAX = 225
M = 16

def to_histogram(x: Tensor):
    x = x * 255
    yuv = torch.matmul(x, torch.tensor([
        [0.29900],
        [0.58700],
        [0.114001]
    ]))
    max_v = torch.max(yuv)
    normalized = yuv / max_v
    scaled = torch.flatten(normalized * (NEW_MAX - 1)).long()
    freq = torch.bincount(scaled, minlength=NEW_MAX)
    cum_histo = torch.cumsum(freq, dim=0)
    normalized_freq = freq / len(scaled)
    print(scaled.shape)
    return normalized_freq * M

image = PIL.Image.open('binary_classification_dataset/valid/12100199_temp.jpg', mode='r').convert('RGB')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.permute(1, 2, 0)),
    transforms.Lambda(to_histogram)
])
plt.figure(figsize=(10,5))

plt.imshow(image)
plt.figure()
# plt.hist(transform(image))
plt.plot(transform(image))
plt.grid(False)
plt.show()