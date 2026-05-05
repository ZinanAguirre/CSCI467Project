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
    
class Conv2(nn.Module):
  def __init__(self, in_channels=3, c1=64, c2=64,
                 fc1=256, fc2=256, out=10, spatial=16):
        super().__init__()
        self.spatial = spatial
 
      
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  
        )
 
       
        fc_in = c2 * spatial * spatial
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(fc_in, fc1),
            nn.ReLU(),
            nn.Linear(fc1, fc2),
            nn.ReLU(),
            nn.Linear(fc2, out),
        )
 
  def forward(self, x):
    x = self.conv_stack(x)
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


def test(dataloader, model, loss_fn, device, Verbose=True):
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
    if Verbose:
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100*correct

def trainEarlyStop(dataloader, val_dataloader, model, loss_fn, optimizer, device, max_iters, patience=5, Verbose=False):
    model.train()
    iteration = 0
    best_val_loss = float('inf')
    best_weights = None
    best_iteration = 0
    checks_without_improvement = 0

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
                model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_iteration = iteration
                    best_weights = copy.deepcopy(model.state_dict())
                    checks_without_improvement = 0
                else:
                    checks_without_improvement += 1

                if Verbose:
                    print(f"iteration: {iteration}, val_loss: {val_loss:>7f}, patience: {checks_without_improvement}/{patience}")

                if checks_without_improvement >= patience:
                    model.load_state_dict(best_weights)
                    return best_iteration

    model.load_state_dict(best_weights)
    return best_iteration


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