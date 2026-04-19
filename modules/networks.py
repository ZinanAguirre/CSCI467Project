import torch
from torch import nn
import copy

class Lenet(nn.Module):
    def __init__(self,ins, l2i, l3i, out):
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