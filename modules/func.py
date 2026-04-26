import torch

def remove(saved, layer, node):
    # remove row from current layer weight
    saved[layer-1][0] = torch.cat([saved[layer-1][0][:node], saved[layer-1][0][node+1:]], dim=0)
    # remove element from current layer bias
    saved[layer-1][1] = torch.cat([saved[layer-1][1][:node], saved[layer-1][1][node+1:]])
    # remove column from next layer weight
    saved[layer][0] = torch.cat([saved[layer][0][:, :node], saved[layer][0][:, node+1:]], dim=1)

def leastweight(saved):
    layer = 0
    node = 0
    v = torch.inf
    for l in range(1,len(saved)):
        importance = saved[l][0].abs().sum(dim=0)
        if v >= min(importance).item():
            v = min(importance).item()
            layer = l
            node = importance.argmin().item()
    return layer, node



def least_magnitude_filters_per_layer(saved, prune_rate=0.10):
  to_remove = []
  for w, b in saved:
    out_C = w.shape[0]
    n_prune = max(1, int(out_C * prune_rate))
    importance = w.abs().sum(dim=(1, 2, 3))  
    indices = importance.argsort()[:n_prune].tolist()
    to_remove.append(sorted(indices, reverse=True))  
  return to_remove

def least_magnitude_neurons_per_layer(saved, prune_rate=0.20):
  to_remove = []
  for w, b in saved[:-1]:          
    out_N = w.shape[0]
    n_prune = max(1, int(out_N * prune_rate))
    importance = w.abs().sum(dim=1) 
    indices = importance.argsort()[:n_prune].tolist()
    to_remove.append(sorted(indices, reverse=True)) 
  return to_remove
