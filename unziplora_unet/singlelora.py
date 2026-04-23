from typing import Optional
import torch
from diffusers.models.lora import LoRACompatibleLinear, LoRALinearLayer
from torch import nn 

from typing import Union

class LoRALinearLayer(nn.Module):
    r"""
    A linear layer that is used with LoRA.

    Parameters:
        in_features (`int`):
            Number of input features.
        out_features (`int`):
            Number of output features.
        rank (`int`, `optional`, defaults to 4):
            The rank of the LoRA layer.
        network_alpha (`float`, `optional`, defaults to `None`):
            The value of the network alpha used for stable learning and preventing underflow. This value has the same
            meaning as the `--network_alpha` option in the kohya-ss trainer script. See
            https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        device (`torch.device`, `optional`, defaults to `None`):
            The device to use for the layer's weights.
        dtype (`torch.dtype`, `optional`, defaults to `None`):
            The dtype to use for the layer's weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor, sigma_mask=None) -> torch.Tensor:
        if sigma_mask is None:
            sigma_mask = torch.ones((1, self.rank), device=hidden_states.device)

        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype)) * sigma_mask
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)
    

class LoRACrossAttnProcessor(nn.Module):
    def __init__(
        self,
        hidden_size,
        lora_linear_layer=LoRALinearLayer,
        cross_attention_dim=None,
        rank=4,
        network_alpha=None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        self.to_q_lora = lora_linear_layer(hidden_size, hidden_size, rank, network_alpha=network_alpha)
        self.to_k_lora = lora_linear_layer(
            cross_attention_dim or hidden_size, hidden_size, rank,network_alpha=network_alpha
        )
        self.to_v_lora = lora_linear_layer(
            cross_attention_dim or hidden_size, hidden_size, rank,network_alpha=network_alpha
        )
        self.to_out_lora = lora_linear_layer(hidden_size, hidden_size, rank,network_alpha=network_alpha)

    def __call__(
        self,
        attn,
        hidden_states, 
        encoder_hidden_states=None, #这个应该是 Text Embedding
        attention_mask=None,
        scale=1.0,
        sigma_mask=None,
    ):
        #B, N, D
        batch_size, sequence_length, _ = hidden_states.shape
        
        # 这里如果 attention_mask 参数是 None 那么最后返回的结果也是 None，
        # 不然的话返回的结果就是: (batch_size * num_heads, 1, sequence_length)
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        query = attn.to_q(hidden_states) + scale * self.to_q_lora(
            hidden_states, sigma_mask
        )

        # query.shape: (batch_size * num_heads, image_tokens, inner_dim)
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )

        
        key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(
            encoder_hidden_states, sigma_mask
        )
        value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(
            encoder_hidden_states, sigma_mask
        )

        # key.shape == value.shape : (batch_size * num_heads, text_tokens, inner_dim)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # attention_probs.shape: (batch_size * num_heads, image_tokens, text_tokens)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        #hidden_states.shape: (batch_size * num_heads, image_tokens, inner_dim)
        hidden_states = torch.bmm(attention_probs, value)
        # hidden_states.shape: (batch_size, num_heads, image_tokens, inner_dim)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj 
        # 通常都不会改变形状
        hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(
            hidden_states, sigma_mask
        )
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        # hidden_states.shape: (batch_size, num_heads, image_tokens, inner_dim)
        return hidden_states
    

