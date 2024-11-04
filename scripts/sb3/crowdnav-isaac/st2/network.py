import torch
from torch import nn
import copy
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

joint_state_dim = 13
MAX_PED_NUM = 8

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Mlp(nn.Module):
    def __init__(self,
                 embed_dim,
                 mlp_ratio,
                 dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class Embedding(nn.Module):
    def __init__(self, joint_state=13, embed_dim=128, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(joint_state, embed_dim)
        nn.init.xavier_normal_(self.embedding.weight)
        nn.init.constant_(self.embedding.bias, 0.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 embed_dim=128,
                 num_heads=8,
                 dropout=0.1,
                 attention_dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_head_size = embed_dim // num_heads
        self.all_head_size = self.attn_head_size * num_heads

        self.qkv = nn.Linear(embed_dim,
                             self.all_head_size * 3)
        nn.init.xavier_normal_(self.qkv.weight)
        nn.init.constant_(self.qkv.bias, 0.0)
        self.scales = self.attn_head_size ** -0.5

        self.out = nn.Linear(self.all_head_size,
                             embed_dim)
        nn.init.xavier_normal_(self.out.weight)
        nn.init.constant_(self.out.bias, 0.0)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(-1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def transpose_multihead(self, x):
        new_shape = list(x.shape[:-1]) + [self.num_heads, self.attn_head_size]
        x = x.reshape(new_shape)
        x = x.permute([0, 2, 1, 3])
        return x

    def forward(self, x):
        size = x.shape

        qkv = self.qkv(x).chunk(3, -1)
        q, k, v = map(self.transpose_multihead, qkv)            # [300, 8, 5, 16]
        k = torch.mean(q, 2, keepdim=True)                      # [300, 8, 1, 16]
        k = k.expand((size[0], 8, size[1], -1))                 # [300, 8, 5, 16]
        k = k.permute([0, 1, 3, 2])                             # [300, 8, 16, 5]
        attn = torch.matmul(q, k)                               # [300, 8, 5, 5]

        # print(attn.shape)
        # attn = q + k
        attn = attn * self.scales
        attn = self.softmax(attn)                               # [300, 8, 5, 5]
        attn = self.attn_dropout(attn)

        z = torch.matmul(attn, v)
        z = z.permute([0, 2, 1, 3])

        new_shape = list(z.shape[:-2]) + [self.all_head_size]
        z = z.reshape(new_shape)
        z = self.out(z)
        z = self.proj_dropout(z)

        return z

class EncoderLayer(nn.Module):
    def __init__(self,
                 embed_dim=128,
                 num_heads=8,
                 mlp_ratio=4.,
                 dropout=0.1,
                 attention_dropout=0.1,
                 drop_path_ratio=0.1):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)

        self.attn = Attention(embed_dim,
                              num_heads,
                              dropout,
                              attention_dropout)

        self.mlp_norm = nn.LayerNorm(embed_dim)

        self.mlp = Mlp(embed_dim, mlp_ratio, dropout)

        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

    def forward(self, x):

        # x = x + self.drop_path(self.attn(self.attn_norm(x)))
        # x = x + self.drop_path(self.mlp(self.mlp_norm(x)))

        h = x
        x = self.attn_norm(x)
        x = self.attn(x)

        x = x + h

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + h

        return x

class Encoder(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 depth,
                 mlp_ratio=4.0,
                 dropout=0.,
                 attention_dropout=0.):
        super().__init__()
        layer_list = []
        for i in range(depth):
            encoder_layer = EncoderLayer(embed_dim,
                                         num_heads,
                                         mlp_ratio=mlp_ratio,
                                         dropout=dropout,
                                         attention_dropout=attention_dropout,)
            layer_list.append(copy.deepcopy(encoder_layer))
        self.layers = nn.ModuleList(layer_list)
        self.encoder_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        out = self.encoder_norm(x)
        return out

class Spatial_Temporal_Transformer(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim:int = 256):
        super(Spatial_Temporal_Transformer, self).__init__(observation_space, features_dim)
        # params setting
        self.joint_state_dim = joint_state_dim
        self.embed_dim = 128
        self.temporal_depth=1
        self.spatial_depth=1
        self.num_heads=8
        self.mlp_ratio=4
        self.dropout=0.1
        self.attention_dropout=0.1
        # create patch embedding with positional embedding
        self.embedding = Embedding(self.joint_state_dim,
                                   self.embed_dim,
                                   self.dropout)
        # create multi head self-attention layers
        self.temporal_encoder = Encoder(self.embed_dim,
                                        self.num_heads,
                                        self.temporal_depth,
                                        self.mlp_ratio,
                                        self.dropout,
                                        self.attention_dropout)
        self.Spatial_encoder = Encoder(self.embed_dim,
                                       self.num_heads,
                                       self.spatial_depth,
                                       self.mlp_ratio,
                                       self.dropout,
                                       self.attention_dropout)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.value_network = nn.Sequential(
            nn.Linear(134, 256),
            nn.GELU(),
            nn.Linear(256, features_dim),
        )
        # self.softmax = nn.Softmax(-1)
        self.attention_weights = None
        
    # observations size:      [env_nums, 3*ped_nums*joint_state]
    def foward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.reshape(-1, 3, ped_num, joint_state_dim)
        b, t, h, w = observations.shape
        robot_state = observations[:, 2:3, 0, :6].reshape(b, 6)
        # spatial_Transformer
        observations = observations.reshape(-1, h, w)
        spatial_state = self.embedding(observations)
        spatial_state = self.Spatial_encoder(spatial_state).reshape(b, t, h, -1)
        # temporal_Transformer
        temporal_state = spatial_state.permute([0, 2, 1, 3]).reshape(b*h, t, -1)
        temporal_state = self.temporal_encoder(temporal_state)
        temporal_state = temporal_state.reshape(b, h, t, -1).permute([0, 2, 1, 3])
        state = temporal_state.reshape(b, t*h, -1)        
        # pooling & concat & value network 
        state = state.permute([0, 2, 1])
        state = self.avgpool(state).flatten(1)
        state = torch.cat((state, robot_state), 1)
        state = self.value_network(state)

        return state
