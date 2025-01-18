## TextPromptIR: Prompting for All-in-One Blind Image Restoration
## Qiuhai Yan, Aiwen Jiang, Kang Chen, Long Peng, Qiaosi Yi, Chenjie Zhang
## https://arxiv.org/abs/2306.13090


import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

from einops import rearrange
import math


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight



class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
##########################################################################
## Channel-attention fusion module
class Channel_attention(nn.Module):
    def __init__(self, dim, gamma=2, b=1):
        super(Channel_attention, self).__init__()
        c = dim
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, embedding):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).contiguous().transpose(-1, -2).contiguous())
        y = y.transpose(-1, -2).contiguous().squeeze(-1).contiguous()
        y = self.sigmoid(y)
        y = embedding * y

        return y


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.fusion = Channel_attention(dim, gamma=2, b=1)

        self.linear1 = nn.Linear(384, 48)
        self.linear1_1 = nn.Linear(48, 48)
        self.linear1_2 = nn.Linear(48, 48)
        self.linear2 = nn.Linear(384, 96)
        self.linear2_1 = nn.Linear(96, 96)
        self.linear2_2 = nn.Linear(96, 96)
        self.linear3 = nn.Linear(384, 192)
        self.linear3_1 = nn.Linear(192, 192)
        self.linear3_2 = nn.Linear(192, 192)
        self.linear7 = nn.Linear(384, 384)
        self.linear7_1 = nn.Linear(384, 384)
        self.linear7_2 = nn.Linear(384, 384)

    def forward(self, x, embedding, num):
        if num == 1:
            embedding = self.linear1(embedding)
        elif num == 2:
            embedding = self.linear2(embedding)
        elif num == 3:
            embedding = self.linear3(embedding)
        elif num == 7:
            embedding = self.linear7(embedding)

        embedding = self.fusion(x, embedding)

        if num == 1:
            embedding1 = self.linear1_1(embedding)
            embedding2 = self.linear1_2(embedding)
        elif num == 2:
            embedding1 = self.linear2_1(embedding)
            embedding2 = self.linear2_2(embedding)
        elif num == 3:
            embedding1 = self.linear3_1(embedding)
            embedding2 = self.linear3_2(embedding)
        elif num == 7:
            embedding1 = self.linear7_1(embedding)
            embedding2 = self.linear7_2(embedding)

        embedding1 = embedding1.unsqueeze(-1).unsqueeze(-1)
        embedding2 = embedding2.unsqueeze(-1).unsqueeze(-1)

        x = x * embedding1
        x = x + embedding2
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)

        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.fusion = Channel_attention(dim, gamma=2, b=1)
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.linear1 = nn.Linear(384, 48)
        self.linear1_1 = nn.Linear(48, 48)
        self.linear1_2 = nn.Linear(48, 48)
        self.linear2 = nn.Linear(384, 96)
        self.linear2_1 = nn.Linear(96, 96)
        self.linear2_2 = nn.Linear(96, 96)
        self.linear3 = nn.Linear(384, 192)
        self.linear3_1 = nn.Linear(192, 192)
        self.linear3_2 = nn.Linear(192, 192)
        self.linear7 = nn.Linear(384, 384)
        self.linear7_1 = nn.Linear(384, 384)
        self.linear7_2 = nn.Linear(384, 384)

    def forward(self, x, embedding, num):
        if num == 1:
            embedding = self.linear1(embedding)
        elif num == 2:
            embedding = self.linear2(embedding)
        elif num == 3:
            embedding = self.linear3(embedding)
        elif num == 7:
            embedding = self.linear7(embedding)

        embedding = self.fusion(x, embedding)

        if num == 1:
            embedding1 = self.linear1_1(embedding)
            embedding2 = self.linear1_2(embedding)
        elif num == 2:
            embedding1 = self.linear2_1(embedding)
            embedding2 = self.linear2_2(embedding)
        elif num == 3:
            embedding1 = self.linear3_1(embedding)
            embedding2 = self.linear3_2(embedding)
        elif num == 7:
            embedding1 = self.linear7_1(embedding)
            embedding2 = self.linear7_2(embedding)

        embedding1 = embedding1.unsqueeze(-1).unsqueeze(-1)
        embedding2 = embedding2.unsqueeze(-1).unsqueeze(-1)
        x = x * embedding1
        x = x + embedding2

        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)

        return out


class FeedForward1(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward1, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention1(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention1, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class resblock(nn.Module):
    def __init__(self, dim):

        super(resblock, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PReLU(),
                                  nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        res = self.body((x))
        res += x
        return res


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x['img'] = x['img'] + self.attn(self.norm1(x['img']), x['emb'], x['num'])
        x['img'] = x['img'] + self.ffn(self.norm2(x['img']), x['emb'], x['num'])

        return x

class TransformerBlock1(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock1, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention1(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward1(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x




##########################################################################
##---------- Prompt Gen Module -----------------------
class PromptGenBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192):
        super(PromptGenBlock, self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size))
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self,x):
        B,C,H,W = x.shape
        emb = x.mean(dim=(-2,-1))
        prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        prompt = torch.sum(prompt,dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt


##########################################################################
##---------- Channel attention module -----------------------
class Prior_Sp(nn.Module):

    def __init__(self, in_dim=32, num=1):
        super(Prior_Sp, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.key_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)

        self.gamma1 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        self.gamma2 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        self.sig = nn.Sigmoid()
        self.num = num
        self.linear1_1 = nn.Linear(192, 48)
        self.linear1_2 = nn.Linear(192, 48)
        self.linear2_1 = nn.Linear(192, 96)
        self.linear2_2 = nn.Linear(192, 96)
        self.linear3_1 = nn.Linear(192, 192)
        self.linear3_2 = nn.Linear(192, 192)

    def forward(self, x, prior, embedding):

        if self.num == 1:
            embedding1 = self.linear1_1(embedding)
            embedding2 = self.linear1_2(embedding)
        elif self.num == 2:
            embedding1 = self.linear2_1(embedding)
            embedding2 = self.linear2_2(embedding)
        elif self.num == 3:
            embedding1 = self.linear3_1(embedding)
            embedding2 = self.linear3_2(embedding)
        embedding1 = embedding1.unsqueeze(-1).unsqueeze(-1)
        embedding2 = embedding2.unsqueeze(-1).unsqueeze(-1)

        x = x + embedding1
        prior = prior + embedding2

        x_q = self.query_conv(x)
        prior_k = self.key_conv(prior)
        energy = x_q * prior_k
        attention = self.sig(energy)
        attention_x = x * attention
        attention_p = prior * attention

        x_gamma = self.gamma1(torch.cat((x, attention_x), dim=1))
        x_out = x * x_gamma[:, [0], :, :] + attention_x * x_gamma[:, [1], :, :]

        p_gamma = self.gamma2(torch.cat((prior, attention_p), dim=1))
        prior_out = prior * p_gamma[:, [0], :, :] + attention_p * p_gamma[:, [1], :, :]

        return x_out, prior_out


##########################################################################
##---------- TextPromptIR -----------------------

class TextPromptIR(nn.Module):
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        dim = 48,
        num_blocks = [4, 6, 6, 8],
        num_refinement_blocks = 4,
        heads = [1, 2, 4, 8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',
    ):

        super(TextPromptIR, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)


        self.prompt1_1 = PromptGenBlock(prompt_dim=32, prompt_len=5, prompt_size=128, lin_dim=48)
        self.prompt1_2 = PromptGenBlock(prompt_dim=64, prompt_len=5, prompt_size=64, lin_dim=96)
        self.prompt2_1 = PromptGenBlock(prompt_dim=64, prompt_len=5, prompt_size=32, lin_dim=96)
        self.prompt2_2 = PromptGenBlock(prompt_dim=64, prompt_len=5, prompt_size=32, lin_dim=96)
        self.prompt3_1 = PromptGenBlock(prompt_dim=128, prompt_len=5, prompt_size=64, lin_dim=192)
        self.prompt3_2 = PromptGenBlock(prompt_dim=128, prompt_len=5, prompt_size=128, lin_dim=192)


        self.noise_level1_1 = TransformerBlock(dim=80, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.noise_level1_2 = TransformerBlock(dim=160, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.noise_level2_1 = TransformerBlock(dim=160, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.noise_level2_2 = TransformerBlock(dim=160, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.noise_level3_1 = TransformerBlock(dim=320, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.noise_level3_2 = TransformerBlock(dim=320, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)


        self.reduce_noise_level1_1 = nn.Conv2d(80, 48, kernel_size=1, bias=bias)
        self.reduce_noise_level1_2 = nn.Conv2d(160, 96, kernel_size=1, bias=bias)
        self.reduce_noise_level2_1 = nn.Conv2d(160, 96, kernel_size=1, bias=bias)
        self.reduce_noise_level2_2 = nn.Conv2d(160, 96, kernel_size=1, bias=bias)
        self.reduce_noise_level3_1 = nn.Conv2d(320, 192, kernel_size=1, bias=bias)
        self.reduce_noise_level3_2 = nn.Conv2d(320, 192, kernel_size=1, bias=bias)

        self.chnl_reduce1 = nn.Conv2d(64,64,kernel_size=1,bias=bias)
        self.chnl_reduce2 = nn.Conv2d(128,128,kernel_size=1,bias=bias)
        self.chnl_reduce3 = nn.Conv2d(320,256,kernel_size=1,bias=bias)

        self.reduce_noise_channel_1 = nn.Conv2d(dim + 64,dim,kernel_size=1,bias=bias)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2

        self.reduce_noise_channel_2 = nn.Conv2d(int(dim*2**1) + 128,int(dim*2**1),kernel_size=1,bias=bias)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3

        self.reduce_noise_channel_3 = nn.Conv2d(int(dim*2**2) + 256,int(dim*2**2),kernel_size=1,bias=bias)
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.noise_level3 = TransformerBlock1(dim=int(dim*2**2) + 512, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.reduce_noise_level3 = nn.Conv2d(int(dim * 2 ** 2) + 512, int(dim * 2 ** 2), kernel_size=1, bias=bias)

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.noise_level2 = TransformerBlock1(dim=int(dim*2**1) + 224, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level2 = nn.Conv2d(int(dim*2**1)+224, int(dim*2**2), kernel_size=1, bias=bias)


        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.noise_level1 = TransformerBlock1(dim=int(dim*2**1)+64, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level1 = nn.Conv2d(int(dim*2**1)+64,int(dim*2**1),kernel_size=1,bias=bias)


        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.linear1 = nn.Linear(768, 576)
        self.LReLU1 = nn.LeakyReLU(0.2)
        self.linear2 = nn.Linear(576, 384)
        self.LReLU2 = nn.LeakyReLU(0.2)


    def forward(self, inp_img, embedding):
        embedding = self.linear1(embedding)
        embedding = self.LReLU1(embedding)
        embedding = self.linear2(embedding)
        embedding = self.LReLU2(embedding)

        inp_enc_level1 = self.patch_embed(inp_img)
        list1 = {'img': inp_enc_level1, 'emb': embedding, 'num': 1}
        out_enc_level1 = self.encoder_level1(list1)
        out_enc_level1 = out_enc_level1['img']


        inp_enc_level2 = self.down1_2(out_enc_level1)
        list2 = {'img': inp_enc_level2, 'emb': embedding, 'num': 2}
        out_enc_level2 = self.encoder_level2(list2)
        out_enc_level2 = out_enc_level2['img']


        inp_enc_level3 = self.down2_3(out_enc_level2)
        list3 = {'img': inp_enc_level3, 'emb': embedding, 'num': 3}
        out_enc_level3 = self.encoder_level3(list3)
        out_enc_level3 = out_enc_level3['img']


        inp_enc_level4 = self.down3_4(out_enc_level3)
        list1 = {'img': inp_enc_level4, 'emb': embedding, 'num': 7}
        latent = self.latent(list1)
        latent = latent['img']
        inp_dec_level3 = self.up4_3(latent)

        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        list3 = {'img': inp_dec_level3, 'emb': embedding, 'num': 3}
        out_dec_level3 = self.decoder_level3(list3)
        out_dec_level3 = out_dec_level3['img']


        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        list2 = {'img': inp_dec_level2, 'emb': embedding, 'num': 2}
        out_dec_level2 = self.decoder_level2(list2)
        out_dec_level2 = out_dec_level2['img']


        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)

        list1 = {'img': inp_dec_level1, 'emb': embedding, 'num': 2}
        out_dec_level1 = self.decoder_level1(list1)
        out_dec_level1 = out_dec_level1['img']


        list1 = {'img': out_dec_level1, 'emb': embedding, 'num': 2}
        out_dec_level1 = self.refinement(list1)
        out_dec_level1 = out_dec_level1['img']
        out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1
