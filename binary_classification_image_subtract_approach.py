import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch 
from torch.utils.data import TensorDataset, DataLoader
import random
from torch import nn
from torch._C._te import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm


TEMP_SUFFIX = "temp"
TEST_SUFFIX = "test"
IMAGE_EXTENSION = ".jpg"
DATASET_LOCATION = '/work/pcb-defect-detection-histogram-approach/binary_classification_dataset'
NOT_VALID_LOCATION = DATASET_LOCATION + "/not_valid"
VALID_LOCATION = DATASET_LOCATION + "/valid"
TOLERANCE = 3
HISTOGRAM_SIZE = 256
TRAIN_VECTOR_SIZE = HISTOGRAM_SIZE * 2
BATCH_SIZE = 32


def get_pcb_image_names(path):
    images = []
    for file in os.listdir(path):
        if file.endswith(IMAGE_EXTENSION):
            images.append(file)
    return images


invalid_pcb_images_names = get_pcb_image_names(Path(NOT_VALID_LOCATION))
print(f"Read PCB images = {invalid_pcb_images_names}")


def to_valid_image_name(name: str):
    return name.replace(TEST_SUFFIX, TEMP_SUFFIX)


def to_cumm_histo(diff_vector):
    # If empty -> return empty histogram (filled with zeros)
    n = len(diff_vector)
    if n == 0:
        return np.zeros(HISTOGRAM_SIZE)
    histo = np.bincount(diff_vector, minlength=HISTOGRAM_SIZE)
    cumm = np.cumsum(histo, axis=0)
    cumm = cumm / n
    return cumm


def find_difference_histograms(img1, img2):
    diff = (np.int32(img1) - np.int32(img2)).flatten()
    significantly_different = diff[np.abs(diff) > TOLERANCE]
    positive = significantly_different[significantly_different > 0]
    negative = significantly_different[significantly_different < 0] * -1
    pos_histo = to_cumm_histo(positive)
    neg_histo = to_cumm_histo(negative)
    return (pos_histo, neg_histo)


def plot_compare_result(img1, img2, histos):
    (pos_histo, neg_histo) = histos
    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(img1)
    axarr[0,1].imshow(img2)
    axarr[1,0].plot(pos_histo)
    axarr[1,1].plot(neg_histo)
    f.show()


def plot_several_examples(num_examples):
    copy_examples = invalid_pcb_images_names.copy()
    random.shuffle(copy_examples)
    for invalid_pcb_image_name in copy_examples:
        if num_examples == 0:
            break
        invalid_pcb_image = cv2.imread(os.path.join(VALID_LOCATION, to_valid_image_name(invalid_pcb_image_name)), cv2.IMREAD_GRAYSCALE)
        valid_pcb_image = cv2.imread(os.path.join(NOT_VALID_LOCATION, invalid_pcb_image_name), cv2.IMREAD_GRAYSCALE)
        histos = find_difference_histograms(invalid_pcb_image, valid_pcb_image)
        plot_compare_result(valid_pcb_image, invalid_pcb_image, histos)
        num_examples -= 1
        # print(f"Original size = {len(diff)} thres = {len(significantly_different)} pos = {len(positive)} neg = {len(negative)}")
        # print(f"Max = {np.min(positive)} / {np.min(negative)}")


# Plot some examples!
plot_several_examples(5)


# Prepare data
X = []
Y = []

N = len(invalid_pcb_images_names)

# Add invalid PCB histograms from images dataset
for invalid_pcb_image_name in invalid_pcb_images_names:
    invalid_pcb_image = cv2.imread(os.path.join(VALID_LOCATION, to_valid_image_name(invalid_pcb_image_name)), cv2.IMREAD_GRAYSCALE)
    valid_pcb_image = cv2.imread(os.path.join(NOT_VALID_LOCATION, invalid_pcb_image_name), cv2.IMREAD_GRAYSCALE)
    (pos_histo, neg_histo) = find_difference_histograms(invalid_pcb_image, valid_pcb_image)
    x = np.concatenate((pos_histo, neg_histo)).flatten()
    X.append(x)
    Y.append(0) # Its defect!

# Add valid PCB histograms
for i in range(N):
    almost_horizontal = (np.random.rand(TRAIN_VECTOR_SIZE, 1) * 0.00001).flatten()
    almost_horizontal_cumm = np.cumsum(almost_horizontal, axis=0)
    X.append(almost_horizontal_cumm)
    Y.append(1)


tensor_X = torch.Tensor(X)
tensor_Y = torch.Tensor(Y)

dataset = TensorDataset(tensor_X, tensor_Y)
train_set, val_set = torch.utils.data.random_split(dataset, [0.90, 0.10])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=TRAIN_VECTOR_SIZE, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


model = NeuralNetwork().to(device)
print(f"Model = {model}")

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

epochs = 150

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    accuracy = (correct / len(y_pred)) * 100
    return accuracy

def plot_metrics(train_losses, train_accuracy, test_losses, test_accuracy):
    plt.figure(figsize=(10,5))
    plt.title("Training and Test Loss")
    plt.plot(list(map(lambda x : x.detach().numpy(), test_losses)), label="test")
    plt.plot(list(map(lambda x : x.detach().numpy(), train_losses)), label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,5))
    plt.title("Training and Test Accuracy")
    plt.plot(test_accuracy, label="test")
    plt.plot(train_accuracy, label="train")
    plt.xlabel("iterations")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

train_losses = []
train_accuracy = []
test_losses = []
test_accuracy = []

for epoch in range(epochs):
    train_accuracy_sum = 0
    count_batches = 0
    sum_loss_train = 0

    for _, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        image, labels = data

        y_logits = model(image).squeeze(1)
        y_pred = torch.round(y_logits)

        # 2. Calculate loss/accuracy
        loss = loss_fn(y_logits, labels.float())
        sum_loss_train += loss
        train_accuracy_sum += accuracy_fn(y_true=labels, y_pred=y_pred)
        count_batches += 1

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backwards
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    acc = train_accuracy_sum / count_batches
    train_loss = sum_loss_train / count_batches

    test_accuracy_sum = 0
    test_count_batches = 0
    sum_loss_test = 0

    model.eval()
    with torch.inference_mode():
        for _, data in tqdm(enumerate(val_loader), total=len(val_loader)):
            image, labels = data
            test_logits = model(image).squeeze(1)
            test_pred = torch.round(torch.sigmoid(test_logits))
            test_loss = loss_fn(test_logits, labels.float())
            sum_loss_test += test_loss
            test_count_batches += 1
            test_accuracy_sum += accuracy_fn(y_true=labels, y_pred=test_pred)

    test_loss_avg = sum_loss_test / test_count_batches
    test_acc = test_accuracy_sum / test_count_batches

    train_losses.append(train_loss)
    train_accuracy.append(acc)
    test_losses.append(test_loss_avg)
    test_accuracy.append(test_acc)

    print(f"Epoch: {epoch} | Loss: {train_loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss_avg:.5f}, Test acc: {test_acc:.2f}%")

plot_metrics(train_losses, train_accuracy, test_losses, test_accuracy)