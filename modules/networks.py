import torch
from torch import nn
import torch.nn.functional as F
import copy

class Lenet(nn.Module):
    def __init__(self, ins, l2i, l3i, out):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(ins, l2i),
            nn.ReLU(),
            nn.Linear(l2i, l3i),
            nn.ReLU(),
            nn.Linear(l3i, out),        
            )

    def forward(self, x):
        x = x.flatten(1)
        return self.linear_relu_stack(x)

class Conv4(nn.Module):
    def __init__(
        self,
        c1=64,
        c2=64,
        c3=128,
        c4=128,
        fc1_size=256,
        fc2_size=256,
        num_classes=10,
        input_size=32,
    ):
        super().__init__()
        # Convolutional Layers
        # CIFAR-10 has 3 input channels (RGB)
        self.conv1 = nn.Conv2d(3, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(c3, c4, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # After two max pools, feature size is reduced by 4 in each spatial dim.
        self.final_spatial_size = input_size // 4
        self.fc1 = nn.Linear(c4 * self.final_spatial_size * self.final_spatial_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(nn.functional.relu(self.conv4(x)))
        
        x = x.flatten(1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100*correct

def trainIter(dataloader, val_dataloader, model, loss_fn, optimizer, device, max_iters, Verbose = False):
    model.train()
    iteration = 0
    val_losses = []
    best_val_loss = float('inf')
    best_weights = None
    best_iteration = 0
    
    while iteration < max_iters:
        for X, y in dataloader:
            if iteration >= max_iters:
                break
                
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            iteration += 1

            if iteration % 100 == 0:
                val_loss = evaluate(val_dataloader, model, loss_fn, device)
                val_losses.append((iteration, val_loss))
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_iteration = iteration
                    best_weights = copy.deepcopy(model.state_dict())
                
                if Verbose == True:
                    print(f"iteration: {iteration}, val_loss: {val_loss:>7f}")
                
                model.train()

    # restore best weights
    model.load_state_dict(best_weights)
    return best_iteration


def evaluate(dataloader, model, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += loss_fn(pred, y).item()
    return total_loss / len(dataloader)