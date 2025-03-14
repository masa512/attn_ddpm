import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.act = nn.Sigmoid()
    def forward(self,q,k,v):
        # Q (Nbatch,Nseq,dk)
        # K (Nbatch,Nseq,dk)
        # V (Nbatch,Nseq,dv)

        qk = q @ k.transpose(-2,-1)
        qk /= self.scale
        qk = self.act(qk)
        return qk @ v

class SelfAttention(nn.Module):

    def __init__(self,emb_dims):

        super().__init__()
        self.qkv = nn.Linear(
            in_features=emb_dims,
            out_features=3 * emb_dims)
        
        self.SDPA = ScaledDotProductAttention()
        
        self.scale = emb_dims ** -0.5
        self.act = nn.Sigmoid()

    def forward(self,x:torch.Tensor):
        # Input is N,C,L # Channel dimension is important (C = 1 for this) for multi head C will be greater

        # Pass through self.qkv and partition (each with B,1,L//3)
        q,k,v = self.qkv(x).chunk(chunks = 3,dim = -1) # Divide output in 3 

        # Apply product and scale and activation (It ignores the batch dimension automatically)
        x = self.SDPA(q,k,v)
        return x


class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self,n_heads,emb_dims,qk_dim,v_dim ):

        super().__init__()
        self.qkv = nn.Linear(
            in_features=emb_dims,
            out_features=3 * emb_dims)
        
        self.pre_mh = nn.Linear(
            in_features=emb_dims * 3,
            out_features=emb_dims * 3 * n_heads,
        )
        
        self.SAUnits = [SelfAttention(emb_dims=emb_dims) for i in range(n_heads)]
    def forward(self,x):

        # Obtain the QKV
        q,k,v = self.qkv(x).chunk(chunks = 3,dim = -1)

        