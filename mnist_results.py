import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch import nn
from torch.utils.data import DataLoader
import copy
from modules import func, networks
import math
from tqdm import tqdm

BATCH_SIZE = 60
MAX_ITERS = 50000 #
LAMBDA = 1.2e-3
HIDDENLAYER1 = 300
HIDDENLAYER2 = 100
OUT = 10
VALSIZE=5000
P = 0.2
ROUNDS = 15 #
NUM_RUNS = 3 #

data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

targets = data.targets.numpy()
idx = list(range(len(data)))

train_idx, val_idx = train_test_split(
    idx,
    test_size=VALSIZE,
    stratify=targets,
    random_state=42,
    shuffle=True
)

train_data = Subset(data, train_idx)
val_data = Subset(data, val_idx)

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

batch_size = BATCH_SIZE

train_dataloader = DataLoader(train_data, batch_size=batch_size)
val_dataloader = DataLoader(val_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

x, y = train_data[0]
inputsize = math.prod(x.shape)

all_iters  = []
all_accs   = []
all_itersR = []
all_accsR  = []

for run in tqdm(range(NUM_RUNS), desc="Runs"):

    model = networks.Lenet(inputsize, HIDDENLAYER1, HIDDENLAYER2, OUT).to(device)
    modelR = networks.Lenet(inputsize, HIDDENLAYER1, HIDDENLAYER2, OUT).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LAMBDA)
    optimizerR = torch.optim.Adam(modelR.parameters(), lr=LAMBDA)



    linear_layers = [(i, m) for i, m in enumerate(model.linear_relu_stack) if isinstance(m, nn.Linear)]
    original = [[layer.weight.data.clone(), layer.bias.data.clone()] for _, layer in linear_layers]

    out = []
    outR = []

    for i in tqdm(range(ROUNDS), desc=f"  Run {run+1} rounds", leave=False):
        bestIter = networks.trainEarlyStop(train_dataloader, val_dataloader, model, loss_fn, optimizer, device, MAX_ITERS)
        a = networks.test(test_dataloader, model, loss_fn, device, Verbose=False)

        bestIterR = networks.trainEarlyStop(train_dataloader, val_dataloader, modelR, loss_fn, optimizerR, device, MAX_ITERS)
        aR = networks.test(test_dataloader, modelR, loss_fn, device, Verbose=False)

        out.append((bestIter, a))
        outR.append((bestIterR, aR))

        layers = [(i, m) for i, m in enumerate(model.linear_relu_stack) if isinstance(m, nn.Linear)]
        saved = [[layer.weight.data.clone(), layer.bias.data.clone()] for _, layer in layers]

        w = func.leastweightRank(saved, 1)
        func.removeNodes(original, w[:int(len(w) * P)])

        hl1 = original[0][0].shape[0]
        hl2 = original[1][0].shape[0]

        modelR = networks.Lenet(inputsize, hl1, hl2, OUT).to(device)
        optimizerR = torch.optim.Adam(modelR.parameters(), lr=LAMBDA)
        
        model = networks.Lenet(inputsize, hl1, hl2, OUT).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LAMBDA)
        new_linear_layers = [(i, m) for i, m in enumerate(model.linear_relu_stack) if isinstance(m, nn.Linear)]

        for (w, b), (_, layer) in zip(original, new_linear_layers):
            layer.weight.data = w.clone()
            layer.bias.data   = b.clone()

    all_iters.append([x[0] for x in out])
    all_accs.append([x[1] for x in out])
    all_itersR.append([x[0] for x in outR])
    all_accsR.append([x[1] for x in outR])

remaining = [100 * (1 - P) ** i for i in range(ROUNDS)]

iters  = [sum(r[i] for r in all_iters)  / NUM_RUNS for i in range(ROUNDS)]
accs   = [sum(r[i] for r in all_accs)   / NUM_RUNS for i in range(ROUNDS)]
itersR = [sum(r[i] for r in all_itersR) / NUM_RUNS for i in range(ROUNDS)]
accsR  = [sum(r[i] for r in all_accsR)  / NUM_RUNS for i in range(ROUNDS)]

xs = list(range(len(remaining)))
labels = [f"{r:.1f}" for r in remaining]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# Row 1: equally spaced
ax1.plot(xs, iters,  label="Pruned (lottery)")
ax1.plot(xs, itersR, label="Random reinit")
ax1.set_xlabel('Nodes Remaining (%)')
ax1.set_ylabel('Best Iteration')
ax1.set_title('Early Stop Iteration vs Nodes Remaining (equal spacing)')
ax1.set_xticks(xs)
ax1.set_xticklabels(labels, rotation=45)
ax1.legend()

ax2.plot(xs, accs,  label="Pruned (lottery)")
ax2.plot(xs, accsR, label="Random reinit")
ax2.set_xlabel('Nodes Remaining (%)')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy vs Nodes Remaining (equal spacing)')
ax2.set_xticks(xs)
ax2.set_xticklabels(labels, rotation=45)
ax2.legend()

# Row 2: actual % spacing
ax3.plot(remaining, iters,  label="Pruned (lottery)")
ax3.plot(remaining, itersR, label="Random reinit")
ax3.set_xlabel('Nodes Remaining (%)')
ax3.set_ylabel('Best Iteration')
ax3.set_title('Early Stop Iteration vs Nodes Remaining')
ax3.set_xticks(remaining)
ax3.set_xticklabels(labels, rotation=45)
ax3.invert_xaxis()
ax3.legend()

ax4.plot(remaining, accs,  label="Pruned (lottery)")
ax4.plot(remaining, accsR, label="Random reinit")
ax4.set_xlabel('Nodes Remaining (%)')
ax4.set_ylabel('Accuracy')
ax4.set_title('Accuracy vs Nodes Remaining')
ax4.set_xticks(remaining)
ax4.set_xticklabels(labels, rotation=45)
ax4.invert_xaxis()
ax4.legend()

plt.tight_layout()
plt.savefig("mnist_results.png", dpi=150, bbox_inches="tight")