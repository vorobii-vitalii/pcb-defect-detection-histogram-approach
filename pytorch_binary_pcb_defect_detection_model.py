import torch
from torch import nn
from torch._C._te import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

BATCH_SIZE = 32
PCB_DATASET_LOCATION = '/work/pcb-defect-detection-histogram-approach/binary_classification_dataset'
NEW_MAX = 225
S = 16


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
    normalized_freq = cum_histo / len(scaled)
    return normalized_freq * S


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.permute(1, 2, 0)),
    transforms.Lambda(to_histogram)
])

dataset = datasets.ImageFolder(PCB_DATASET_LOCATION, transform=transform)
train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])


train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

# Choose device

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=NEW_MAX, out_features=1),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# torch.manual_seed(42)

epochs = 15


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()  # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc


for epoch in range(epochs):
    train_accuracy_sum = 0
    count_batches = 0
    sum_loss_train = 0

    for _, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        image, labels = data

        y_logits = model(image).squeeze(1)
        y_pred = torch.round(torch.sigmoid(y_logits))  # logits -> predicition probabilities -> prediction labels

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

    print(f"Epoch: {epoch} | Loss: {train_loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss_avg:.5f}, Test acc: {test_acc:.2f}%")
