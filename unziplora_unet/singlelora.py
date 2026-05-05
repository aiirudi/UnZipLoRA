from typing import Dict, Optional
import torch
from diffusers.models.lora import LoRACompatibleLinear, LoRALinearLayer

from torch import nn 
import torch.nn.functional as F

from typing import Union,ClassVar,Optional

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

    _active_mask_content: ClassVar[Optional[torch.Tensor]] = None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        use_base_weight: bool = False,
        use_mask: bool = False,
    ):
        super().__init__()
        # 加入 TFM 的 mask
        self.use_mask = use_mask

        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)

        self.use_base_weight = use_base_weight
        if use_base_weight:
            self.register_buffer('base_down', torch.zeros(rank,in_features, device=device, dtype=dtype))
            self.register_buffer('base_up', torch.zeros(out_features,rank, device=device, dtype=dtype))

        self.network_alpha = network_alpha
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    @classmethod
    def set_content_mask(cls, mask: Optional[torch.Tensor]) -> None:
        cls._active_mask_content = mask

    def forward(self, hidden_states: torch.Tensor, sigma_mask=None) -> torch.Tensor:
        if sigma_mask is None:
            # 警告：如果 sigma_mask 为 None，使用全秩（这可能不是期望的行为）
            # 在推理时应该通过 cross_attention_kwargs 传递 sigma_mask
            sigma_mask = torch.ones((1, self.rank), device=hidden_states.device, dtype=self.down.weight.dtype)

        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype)) * sigma_mask
        up_hidden_states = self.up(down_hidden_states)

        if self.use_base_weight:
            base_down_out = F.linear(hidden_states.to(dtype), self.base_down) * sigma_mask
            base_up_out = F.linear(base_down_out, self.base_up)
            up_hidden_states = up_hidden_states - base_up_out
        
        if self.use_mask and LoRALinearLayer._active_mask_content is not None:
            mask = LoRALinearLayer._active_mask_content.to(up_hidden_states.dtype)
            up_hidden_states = up_hidden_states * mask

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
        use_base_weight: bool = False,
        use_mask: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        self.to_q_lora = lora_linear_layer(
            hidden_size,
            hidden_size,
            rank,
            network_alpha=network_alpha,
            use_base_weight=use_base_weight,
            use_mask=False,
        )
        self.to_k_lora = lora_linear_layer(
            cross_attention_dim or hidden_size,
            hidden_size,
            rank,
            network_alpha=network_alpha,
            use_base_weight=use_base_weight,
            use_mask=use_mask,
        )
        self.to_v_lora = lora_linear_layer(
            cross_attention_dim or hidden_size,
            hidden_size,
            rank,
            network_alpha=network_alpha,
            use_base_weight=use_base_weight,
            use_mask=use_mask,
        )
        self.to_out_lora = lora_linear_layer(
            hidden_size,
            hidden_size,
            rank,
            network_alpha=network_alpha,
            use_base_weight=use_base_weight,
            use_mask=False,
        )

    def _load_single_lora_layer(
        self,
        layer: LoRALinearLayer,
        down_weight: Optional[torch.Tensor],
        up_weight: Optional[torch.Tensor],
        base_down_weight: Optional[torch.Tensor] = None,
        base_up_weight: Optional[torch.Tensor] = None,
    ) -> None:
        if down_weight is not None and up_weight is not None:
            layer.down.weight.data = down_weight.to(layer.down.weight.device, dtype=layer.down.weight.dtype)
            layer.up.weight.data = up_weight.to(layer.up.weight.device, dtype=layer.up.weight.dtype)

        if not layer.use_base_weight:
            return

        if base_down_weight is not None and base_up_weight is not None:
            layer.base_down.data = base_down_weight.to(layer.base_down.device, dtype=layer.base_down.dtype)
            layer.base_up.data = base_up_weight.to(layer.base_up.device, dtype=layer.base_up.dtype)

    def load_from_state_dicts(
        self,
        *,
        lora_prefix: str,
        base_prefix: str,
        lora_state_dict: Dict[str, torch.Tensor],
        base_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        branch: Optional[str] = None,
    ) -> None:
        part_to_layer = {
            "to_q": self.to_q_lora,
            "to_k": self.to_k_lora,
            "to_v": self.to_v_lora,
            "to_out.0": self.to_out_lora,
        }
        # 现在在自定义 pipeline 中的 prefix 前缀应该没问题
        # prefix: unet.unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1 这里符合 lora_state_dict 的键

        if base_state_dict is not None and branch is None:
            raise ValueError("`branch` must be provided when `base_state_dict` is passed.")

        for part, layer in part_to_layer.items():
            down_weight = lora_state_dict.get(f"{lora_prefix}.{part}.lora.down.weight")
            up_weight = lora_state_dict.get(f"{lora_prefix}.{part}.lora.up.weight")

            down_name = f"{lora_prefix}.{part}.lora.down.weight"
            up_name = f"{lora_prefix}.{part}.lora.up.weight"
            
            """
            print(f'lora_down_weight: {down_name in lora_state_dict}')
            print(f'lora_up_weight: {up_name in lora_state_dict}')
            print()
            """
            
            base_down_weight = None
            base_up_weight = None
            if base_state_dict is not None and branch is not None:
                base_down_weight = base_state_dict.get(f"{base_prefix}.{part}.lora.down.base_{branch}")
                base_up_weight = base_state_dict.get(f"{base_prefix}.{part}.lora.up.base_{branch}")

                base_down_name = f"{base_prefix}.{part}.lora.down.base_{branch}"
                base_up_name = f"{base_prefix}.{part}.lora.up.base_{branch}"
                
                """
                print(f'base_down_weight: {base_down_name in base_state_dict}')
                print(f'base_up_weight: {base_up_name in base_state_dict}')
                print(base_down_name)
                print(base_up_name)
                print()
                """

            self._load_single_lora_layer(
                layer=layer,
                down_weight=down_weight,
                up_weight=up_weight,
                base_down_weight=base_down_weight,
                base_up_weight=base_up_weight,
            )

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None, #这个应该是 Text Embedding
        attention_mask=None,
        scale=1.0,
        **kwargs,  # 接收 cross_attention_kwargs 中的额外参数
    ):
        # 从 kwargs 中提取 sigma_mask
        sigma_mask = kwargs.get("sigma_mask", None)

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
    
