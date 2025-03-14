import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):

    def __init__(self,dk:int,dv:int):
        super().__init__()
        self.dk = dk
        self.dv = dv
        self.act = nn.Softmax(-1)
        self.scale = dk ** -0.5
    def forward(self,q:torch.Tensor,k:torch.Tensor,v:torch.Tensor):
        # Q (B,n_heads,N,dk)
        # K (B,n_heads,N,dk)
        # V (B,n_heads,N,dv)

        qk = q @ k.transpose(-2,-1)
        qk /= self.scale
        qk = self.act(qk)

        return qk @ v

class SingleHeadSelfAttention(nn.Module):

    def __init__(self,emb_dims:int,dk:int,dv:int):

        super().__init__()

        self.emb_dims = emb_dims
        self.dk = dk
        self.dv = dv

        self.qkv = nn.Linear(
            in_features=emb_dims,
            out_features=2 * self.dk + self.dv)
        
        self.SDPA = ScaledDotProductAttention(dk=self.dk,dv=self.dv)
        
        self.scale = self.dk ** -0.5
        self.act = nn.Softmax(-1)

        self.o = nn.Linear(
            in_features=dv,
            out_features=emb_dims)

    def forward(self,x:torch.Tensor):
        # Input is Nbatch by Nseq by emb_dim
        # Add n_head dimension

        assert x.dim() == 3 or x.dim() == 4, "Wrong place"
        if x.dim() == 3:
            x = x.unsqueeze(dim=1)

        # Pass through self.qkv and partition
        q,k,v = self.qkv(x).split([self.dk,self.dk,self.dv],dim = -1) # Divide output in 3 

        # Apply product and scale and activation (It ignores the batch dimension automatically)
        x = self.SDPA(q,k,v)
        return self.o(x).squeeze()


class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self,n_heads:int,emb_dims:int,dk:int,dv:int):

        super().__init__()
        assert dk%n_heads == 0 and dv%n_heads == 0, "The key and value dimensions dont add up to be divisible by n_heads"

        self.n_heads = n_heads
        self.emb_dims = emb_dims
        self.dk = dk
        self.dv = dv
        self.scale = dk**-0.5
        self.activation = nn.Softmax(-1)

        self.qkv = nn.Linear(
            in_features=emb_dims,
            out_features= n_heads * (2 * dk + dv))
        
        self.SDPA = ScaledDotProductAttention(dk=self.dk,dv=self.dv)

        self.o = nn.Linear(
            in_features = n_heads * dv,
            out_features = emb_dims
        )

    def forward(self,x:torch.Tensor):

        
        # Obtain the QKV as a package
        B = x.size(0)
        N = x.size(1)

        qkv = self.qkv(x)# Each will have dimensions B,N,2*dk+dv
        
        # Reshape to partition in to B,n_head,N,(2*dk + dv)/n_head
        qkv = qkv.contiguous().view(B,self.n_heads,N,-1)

        # Extract Q,K, and V
        q,k,v = torch.split(qkv,[self.dk,self.dk,self.dv],dim = -1)

        # Scaled Dot Product Attention
        y = self.SDPA(q,k,v)
        
        y = y.contiguous().view(B,N,-1)

        # Apply output layer
        y = self.o(y)

        return y


    