import torch

def remove(saved, layer, node):
    # remove row from current layer weight
    saved[layer][0] = torch.cat([saved[layer][0][:node], saved[layer][0][node+1:]], dim=0)
    # remove element from current layer bias
    saved[layer][1] = torch.cat([saved[layer][1][:node], saved[layer][1][node+1:]])
    # remove column from next layer weight
    saved[layer+1][0] = torch.cat([saved[layer+1][0][:, :node], saved[layer+1][0][:, node+1:]], dim=1)

def leastweight(saved):
    layer = 0
    node = 0
    v = torch.inf
    for l in range(len(saved)-1):
        importance = saved[l][0].abs().sum(dim=0)
        if v >= min(importance).item():
            v = min(importance).item()
            layer = l
            node = importance.argmin().item()
    return layer, node