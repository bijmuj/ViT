import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PreNorm(nn.Module):
    """LayerNorm done before Attention and Linear layers."""    

    def __init__(self, dim, fn):
        """Constructor

        Args:
            dim (int): Passed to LayerNorm. Normalizes over dimension of this size. 
            fn (function): Function to pass normalized input to.
        """        
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x):
        return self.fn(self.norm(x))


class Residual(nn.Module):
    """Wrapper around Skip Connections.""" 

    def __init__(self, fn):
        """Constructor

        Args:
            fn (function): Function around which skip connection occurs.
        """        
        super().__init__()
        self.fn = fn
    
    def forward(self, x):
        return x + self.fn(x)


class Mlp(nn.Module):
    """Mlp block in Transformer Encoder."""    

    def __init__(self, dim, mlp_hidden, dropout=0.):
        """Constructor

        Args:
            dim (int): Hidden size D of the Attention layer(d_model).
            mlp_hidden (int): Size of the hidden layer in the MLP block.
            dropout (int, optional): Strength of dropout layer. Higher values lead to greater likelihood of inputs getting dropped.
                    Defaults to 0..
        """        
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(dim, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    """Attention Layer of Transformer Encoder."""    

    def __init__(self, dim, heads=12, dropout=0.):
        """Constructor

        Args:
            dim (int): Hidden size D(d_model). The multiheaded attention basically produces this number of outputs per input.
            heads (int, optional): Number of heads used for multiheaded attention. Defaults to 12.
            dropout ([type], optional): Strength of dropout layer. Higher values lead to greater likelihood of inputs getting dropped.
                    Defaults to 0..
        """        
        super().__init__()
        self.heads = heads
        self.d_k = dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias = False)
        self.attention = nn.Softmax(dim=-1)
        self.fc = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Transform the input into query, key and value 
        b, n, _, h = *x.shape, self.heads
        qkv = self.qkv(x).chunk(3, dim = -1)
        q = rearrange(qkv[0], 'b n (h d) -> b h n d', h=h)
        k = rearrange(qkv[1], 'b n (h d) -> b h n d', h=h)
        v = rearrange(qkv[2], 'b n (h d) -> b h n d', h=h)

        # Calculate attention scores and probabilities from query and key
        attention = torch.matmul(q, k.transpose(-1, -2))
        attention = self.attention(attention)

        # Getting context using values and attention scores and reshaping
        out = torch.matmul(attention, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        # Final pass through a linear transformation and dropout
        out = self.fc(out)
        out = self.dropout(out)
        return out


class Transformer(nn.Module):
    """The Transformer Encoder itself."""

    def __init__(self, depth, dim, heads, attention_dropout, dropout, mlp_hidden):
        """Constructor

        Args:
            depth (int): Number of Encoders.
            dim (int): Hidden size D(d_model).
            heads (int): Number of heads.
            dropout (int): Strength of dropout.
            mlp_hidden (int): Size of the hidden layer of the mlp block.
        """        
        super().__init__()
        self.net  = nn.ModuleList([])
        for _ in range(depth):
            self.net.append(nn.ModuleList([
                    Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=attention_dropout))),
                    Residual(PreNorm(dim, Mlp(dim, mlp_hidden, dropout=dropout)))
            ]))
    
    def forward(self, x):
        for att, mlp in self.net:
            x = att(x)
            x = mlp(x)
        return x

class ViT(nn.Module):
    """Vision Transformer."""    

    def __init__(self, image_size=224, patch_size=16, n_classes=100, depth=12, 
            dim=768, heads=12, attention_dropout=0., dropout=0., mlp_hidden=3072):
        """Constructor. The default args are the configuration called "ViT-B/16" in the paper. 
        Except for n_classes, which is set to 100 for CIFAR100 instead of 1000.

        Args:
            image_size (int, optional): Size of image(height=width). Defaults to 224.
            patch_size (int, optional): Size of each patch. Defaults to 16.
            n_classes (int, optional): Number of classes for classification. Defaults to 100.
            depth (int, optional): Depth of the Transformer Encoder. Defaults to 12.
            dim (int, optional): Dimension of the multiheaded attention block(d_model). Defaults to 768.
            heads (int, optional): Number of heads. Defaults to 12.
            dropout ([type], optional): Strength of dropout. Defaults to 0..
            mlp_hidden (int, optional): Size of hidden layer of mlp block in Transformer Encoder. Defaults to 3072.
        """        
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        # Splits the image into patches of (patch_size, patch_size) shape and produces a linear projection of it.
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = Transformer(depth, dim, heads, attention_dropout, dropout, mlp_hidden)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_classes)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        # Appending a class token to the beginning of the embedded input.
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        # Adding position embeddings to each of the input patches.
        x += self.pos_embedding[:, :(n + 1)]

        x = self.transformer(x)
        x = x[:, 0]

        return self.mlp_head(x)