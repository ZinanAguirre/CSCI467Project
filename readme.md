## Structural Lottery Tickets: Node Pruning for Efficient Neural Networks

Investigates whether the lottery ticket hypothesis extends to structural pruning — removing entire neurons rather than masking individual weights — to produce genuinely smaller, faster networks without specialized sparse hardware support.

Requirements are in `requirements.txt`. Python version 3.11.5.

### Setup
```
pip install -r requirements.txt
```

---

### Lenet on MNIST (Tyler Davis)
- Fully connected network: 784 → 300 → 100 → 10
- Iterative structural pruning: 15 rounds at 20% per round
- Prunes all layers together
- Nodes ranked by mean absolute outgoing weight across all hidden layers
- Early stopping on validation set (patience=5, max 50,000 iterations)
- Compares lottery ticket rewind vs random reinitialization at each pruning level
- Results averaged over 3 independent runs
- Run: `python mnist_results.py`
- Output saved to `mnist_results.png`

---

### CONV2 on CIFAR-10 (Zinan Aguirre)
- CONV2 convolutional network on CIFAR-10 — architecture specs in `models.png`
- Prunes each layer seperately

---

### CONV4 on CIFAR-10 (Emmanuel Ezirim)
- CONV4 convolutional network on CIFAR-10

---

### Project Structure
```
modules/
  networks.py   — model definitions and training functions
  func.py       — pruning and node ranking utilities
mnist_results.py — Lenet MNIST experiment
requirements.txt
```