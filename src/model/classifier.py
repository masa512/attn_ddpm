from .attn_module import SingleHeadSelfAttention,MultiHeadSelfAttention
import torch
import torch.nn as nn


class BaselineClassifer(nn.Module):

    def __init__(self,width,height,emb_dim,mid_dim,n_classes):

        super().__init__()

        self.emb_layer = nn.Linear(width*height,emb_dim)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim,mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim,n_classes)
        )

    def forward(self,x):
        
        N = x.size(0)
        x = x.contiguous().view(N,-1)
        x = self.emb_layer(x)
        x = self.classifier(x)

        return x

class AttnClassifier(nn.Module):

    def __init__(self,width,height,emb_dim,mid_dim,n_classes,n_heads,dk,dv):
        super().__init__()
        self.emb_layer = nn.Linear(width*height,emb_dim)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim,mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim,n_classes)
        )

        self.MHSA = MultiHeadSelfAttention(n_heads,emb_dim,dk,dv)

    def forward(self,x):

        N = x.size(0)
        x = x.contiguous().view(N,-1)
        x = self.emb_layer(x)
        # reshape 
        x = x.contiguous().view(N,1,-1)
        # feed to the MHSA
        x = self.MHSA(x).squeeze()
        # feed to output layers
        return self.classifier(x)



