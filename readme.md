## Update

### MNIST (Lenet)
mnist.ipynb is working. Implemented the lottery ticket experiment and tracked 
accuracy and best iteration (early stop point) across pruning rounds. Haven't 
run for the full iteration count yet but plots look correct.

### Next Steps: CONV2 (CIFAR10)
- CONV2 uses CIFAR10 instead of MNIST — architecture specs are in models.png
- Global variables are defined and can be changed for easy configuration
- Will need a new class in networks.py for CONV2 (and any future architectures)

### Implementation Notes
- Used iteration-based training (not epoch-based) as in the paper
- Early stopping is done retroactively — train for all iterations, then find 
  the iteration of minimum validation loss
- Need to benchmark against randomly reinitialized weights at the same size 
  and run a hypothesis test to validate results

### Python
- requirements in txt file
- Python version is 3.11.5