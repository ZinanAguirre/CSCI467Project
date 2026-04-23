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


def leastweight_4d(saved):
    layer_idx = 0
    node_idx = 0
    v = torch.inf
    
    for l in range(len(saved) - 1):
        weight_tensor = saved[l][0]
        
        # If it's a Linear layer (2D: [out, in])
        if len(weight_tensor.shape) == 2:
            importance = weight_tensor.abs().sum(dim=1) 
            
        # If it's a Conv layer (4D: [out, in, H, W])
        elif len(weight_tensor.shape) == 4:
            importance = weight_tensor.abs().sum(dim=(1, 2, 3)) 
        else:
            continue
            
        if v >= min(importance).item():
            v = min(importance).item()
            layer_idx = l
            node_idx = importance.argmin().item()
            
    return layer_idx, node_idx


def remove_4d(saved, layer, node):
    """
    Removes a filter (node) from a Conv2d layer and updates the next layer's inputs.
    """
    # 1. Get the original number of out_channels BEFORE we alter it. 
    # We need this to calculate the flatten chunk size if the next layer is Linear.
    current_out_channels = saved[layer][0].shape[0]
    next_weight = saved[layer+1][0]
    
    # 2. Alter the CURRENT layer (Remove the filter and its bias)
    # dim=0 is the out_channels dimension for Conv2d
    saved[layer][0] = torch.cat([saved[layer][0][:node], saved[layer][0][node+1:]], dim=0)
    saved[layer][1] = torch.cat([saved[layer][1][:node], saved[layer][1][node+1:]], dim=0)
    
    # 3. Alter the NEXT layer (Remove the corresponding inputs)
    if len(next_weight.shape) == 4:
        # The next layer is ANOTHER CONV LAYER
        # dim=1 is the in_channels dimension for Conv2d
        saved[layer+1][0] = torch.cat([next_weight[:, :node], next_weight[:, node+1:]], dim=1)
        
    elif len(next_weight.shape) == 2:
        # The next layer is a LINEAR LAYER (The transition boundary)
        # We must remove a "chunk" of columns corresponding to the flattened spatial size.
        # chunk_size = total input features / number of channels
        if next_weight.shape[1] % current_out_channels != 0:
            raise ValueError(
                "Linear input size is not divisible by previous conv out_channels. "
                "Cannot infer flattened chunk for structured conv pruning."
            )
        chunk_size = next_weight.shape[1] // current_out_channels
        
        # Calculate exactly which columns belong to the removed filter
        start_col = node * chunk_size
        end_col = (node + 1) * chunk_size
        
        # Concatenate everything before the chunk and everything after the chunk
        saved[layer+1][0] = torch.cat([next_weight[:, :start_col], next_weight[:, end_col:]], dim=1)


def least_conv_filters(saved):
    """
    Return all conv filters ranked by ascending absolute-weight importance.
    Each entry is (layer_index, filter_index, importance_value).
    """
    ranked = []
    for layer_idx in range(len(saved) - 1):
        weight_tensor = saved[layer_idx][0]
        if len(weight_tensor.shape) != 4:
            continue
        importance = weight_tensor.abs().sum(dim=(1, 2, 3))
        for filter_idx, imp in enumerate(importance):
            ranked.append((layer_idx, filter_idx, imp.item()))
    ranked.sort(key=lambda x: x[2])
    return ranked