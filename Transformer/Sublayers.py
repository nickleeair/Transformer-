import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    #softmax results
    scores = F.softmax(scores, dim=-1)
    
    # pruning
    pruning_ratio = 0.1  # hyperparameter
    pruning_ratio = 1 - pruning_ratio
    num_columns = scores.shape[3]  # get column number
    num_columns_to_stay = int(num_columns * pruning_ratio)
    #print(num_columns_to_prune)

    column_sum = torch.sum(scores, dim=2, keepdim = True)  # get sum of column elements
    batch_size = scores.shape[0]
    num_heads = scores.shape[1]
    print("column_sum size ")
    print(column_sum.shape)

    # Reshape column_sum to (batch_size * num_heads, column)
    

    # Get the index of the minimum sum along the column dimension (dim=1) within each group
    if scores.shape[3]>num_columns_to_stay:
        min_sum_value, min_sum_indices = torch.topk(column_sum, k=num_columns_to_stay, dim=3)
        #print(f"min_sum_indices = {min_sum_indices}")
        print("min_sum_indices size")
        print(min_sum_indices.shape)

    #topk_sum_value, topk_sum_indices = torch.topk(reshaped_column_sum, k=, dim=1)
    #print("topk_sum_indices")
    #print(topk_sum_indices)

    # Reshape min_sum_indices back to (batch_size, num_heads)
    #min_sum_indices = min_sum_indices.view(batch_size, num_heads)
    #print(min_sum_indices.shape)
    
    #print("min_sum_indices") 
    #print(min_sum_indices)
    
    # Create the pruning mask
    prune_mask = torch.zeros_like(scores)
    
    # Set the corresponding columns in the mask to zero
    for i in range(min_sum_indices.shape[0]):
        for j in range(min_sum_indices.shape[1]):
            for k in range(min_sum_indices.shape[3]):
                prune_mask[i, j, :, min_sum_indices[i, j, 0, k]] = 1

    # Apply the mask to prune the scores
    pruned_scores = scores * prune_mask

    #print("pruned_scores shape")
    #print(pruned_scores.shape)
    #print("scores shape")
    #print(scores.shape)
    #print("prune_mask")
    #print(prune_mask)

    #print("pruned_scores")
    #print(pruned_scores)

    if dropout is not None:
        scores = dropout(pruned_scores)
        #scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        output = self.out(concat)
    
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
    
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
