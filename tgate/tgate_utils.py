import torch
import torch.nn.functional as F
from diffusers.image_processor import IPAdapterMaskProcessor
from diffusers.utils import (
    BACKENDS_MAPPING,
    deprecate,
    logging
)
import inspect
from typing import Callable, List, Optional, Tuple, Union
logger = logging.get_logger(__name__) 

def register_forward(
    model, 
    filter_name: str = 'Attention', 
    keep_shape: bool = True,
    sa_kward: dict = None,
    ca_kward: dict = None,
    **kwargs,
):
    """
    A customized forward function for cross attention layer. 
    Detailed information in https://github.com/HaozheLiu-ST/T-GATE

    Args:
        model (`torch.nn.Module`):
            A diffusion model contains cross attention layers.
        filter_name (`str`):
            The name to filter the selected layer.
        keep_shape (`bool`):
            Whether or not to remain the shape of hidden features
        sa_kward: (`dict`):
            A kwargs dictionary to pass along to the self attention for caching and reusing. 
        ca_kward: (`dict`):
            A kwargs dictionary to pass along to the cross attention for caching and reusing. 

    Returns:
        count (`int`): The number of the cross attention layers used in the given model.
    """

    count = 0
    def warp_custom(
        self: torch.nn.Module,
        keep_shape:bool = True,
        ca_kward: dict = None,
        sa_kward: dict = None,
        **kwargs,
    ):
        def forward(
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            keep_shape = keep_shape,
            ca_cache = ca_kward['cache'],
            sa_cache = sa_kward['cache'],
            ca_reuse = ca_kward['reuse'],
            sa_reuse = sa_kward['reuse'],
            **cross_attention_kwargs,
        ) -> torch.Tensor:
            r"""
            The forward method of the `Attention` class.

            Args:
                hidden_states (`torch.Tensor`):
                    The hidden states of the query.
                encoder_hidden_states (`torch.Tensor`, *optional*):
                    The hidden states of the encoder.
                attention_mask (`torch.Tensor`, *optional*):
                    The attention mask to use. If `None`, no mask is applied.
                **cross_attention_kwargs:
                    Additional keyword arguments to pass along to the cross attention.

            Returns:
                `torch.Tensor`: The output of the attention layer.
            """
            # The `Attention` class can call different attention processors / attention functions
            # here we simply pass along all tensors to the selected processor class
            # For standard processors that are defined here, `**cross_attention_kwargs` is empty

            if not hasattr(self,'cache'):
                self.cache = None
            attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
            quiet_attn_parameters = {"ip_adapter_masks"}
            unused_kwargs = [
                k
                for k, _ in cross_attention_kwargs.items()
                if k not in attn_parameters and k not in quiet_attn_parameters
            ]
            
            if len(unused_kwargs) > 0:
                logger.warning(
                    f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
                )
            
            cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}
            
            hidden_states, cache =  tgate_processor(
                self,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                keep_shape = keep_shape,
                cache = self.cache,
                ca_cache = ca_cache,
                sa_cache = sa_cache,
                ca_reuse = ca_reuse,
                sa_reuse = sa_reuse,
                **cross_attention_kwargs,
            )
            if cache != None:
                self.cache = cache
            return hidden_states
        return forward

    def register_recr(
        net: torch.nn.Module, 
        count: int = None, 
        keep_shape:bool = True, 
        ca_kward:dict = None,
        sa_kward:dict = None
    ):
        if net.__class__.__name__ == filter_name:
            net.forward = warp_custom(net, keep_shape = keep_shape, ca_kward = ca_kward,sa_kward = sa_kward)
            return count + 1
        elif hasattr(net, 'children'):
            for net_child in net.children():
                count = register_recr(net_child, count, keep_shape = keep_shape, ca_kward = ca_kward,sa_kward = sa_kward)
        return count

    return register_recr(model, count, keep_shape = keep_shape, ca_kward = ca_kward,sa_kward = sa_kward) 


def tgate_processor(
    attn = None,
    hidden_states = None,
    encoder_hidden_states = None,
    attention_mask  = None,
    temb  = None,
    cache = None,
    keep_shape = True,
    ca_cache = False,
    sa_cache = False,
    ca_reuse = False,
    sa_reuse = False,
    ip_adapter_masks: Optional[torch.Tensor] = None,
    *args,
    **kwargs,
) -> torch.FloatTensor:

    r"""
    A customized forward function of the `IPAdapterAttnProcessor2_0` class.

    Args:
        hidden_states (`torch.Tensor`):
            The hidden states of the query.
        encoder_hidden_states (`torch.Tensor`, *optional*):
            The hidden states of the encoder.
        attention_mask (`torch.Tensor`, *optional*):
            The attention mask to use. If `None`, no mask is applied.
        ip_adapter_masks (`torch.Tensor`, *optional*): 
            The IP adapter masks to use.
        **cross_attention_kwargs:
            Additional keyword arguments to pass along to the cross attention.

    Returns:
        `torch.Tensor`: The output of the attention layer.
    """

    if not hasattr(F, "scaled_dot_product_attention"):
        raise ImportError("IPAdapterAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    if len(args) > 0 or kwargs.get("scale", None) is not None:
        deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
        deprecate("scale", "1.0.0", deprecation_message)

    residual = hidden_states

    # separate ip_hidden_states from encoder_hidden_states
    if encoder_hidden_states is not None:
        if isinstance(encoder_hidden_states, tuple):
            encoder_hidden_states, ip_hidden_states = encoder_hidden_states
        else:
            deprecation_message = (
                "You have passed a tensor as `encoder_hidden_states`. This is"
                " deprecated and will be removed in a future release. Please make sure"
                " to update your script to pass `encoder_hidden_states` as a tuple to"
                " suppress this warning."
            )
            deprecate(
                "encoder_hidden_states not a tuple",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            end_pos = encoder_hidden_states.shape[1] - attn.num_tokens[0]
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                [encoder_hidden_states[:, end_pos:, :]],
            )
    cross_attn = encoder_hidden_states is not None
    self_attn =  encoder_hidden_states is None

    input_ndim = hidden_states.ndim

    # if the attention is cross-attention or self-attention
    if cross_attn and ca_reuse and cache != None:
        hidden_states = cache
    elif self_attn and sa_reuse and cache != None:
        hidden_states = cache
    else:

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if ip_adapter_masks is not None:
            if not isinstance(ip_adapter_masks, List):
                # for backward compatibility, we accept `ip_adapter_mask` as a tensor
                # of shape [num_ip_adapter, 1, height, width]
                ip_adapter_masks = list(ip_adapter_masks.unsqueeze(1))
            if not (len(ip_adapter_masks) == len(attn.scale) == len(ip_hidden_states)):
                raise ValueError(
                    f"Length of ip_adapter_masks array ({len(ip_adapter_masks)}) must"
                    f" match length of self.scale array ({len(attn.scale)}) and number"
                    f" of ip_hidden_states ({len(ip_hidden_states)})"
                )
            else:
                for index, (mask, _scale, ip_state) in enumerate(
                    zip(ip_adapter_masks, attn.scale, ip_hidden_states)
                ):
                    if not isinstance(mask, torch.Tensor) or mask.ndim != 4:
                        raise ValueError(
                            "Each element of the ip_adapter_masks array should be a"
                            " tensor with shape [1, num_images_for_ip_adapter, height,"
                            " width]. Please use `IPAdapterMaskProcessor` to preprocess"
                            " your mask"
                        )
                    if mask.shape[1] != ip_state.shape[1]:
                        raise ValueError(
                            f"Number of masks ({mask.shape[1]}) does not match number"
                            f" of ip images ({ip_state.shape[1]}) at index {index}"
                        )
                    if isinstance(_scale, list) and len(_scale) != mask.shape[1]:
                        raise ValueError(
                            f"Number of masks ({mask.shape[1]}) does not match "
                            f"number of scales ({len(_scale)}) at index {index}"
                        )
        else:
            ip_adapter_masks = [None] * len(attn.scale)

        # for ip-adapter
        for current_ip_hidden_states, _scale, to_k_ip, to_v_ip, mask in zip(
            ip_hidden_states, attn.scale, attn.to_k_ip, attn.to_v_ip, ip_adapter_masks
        ):
            skip = False
            if isinstance(_scale, list):
                if all(s == 0 for s in _scale):
                    skip = True
            elif _scale == 0:
                skip = True
            if not skip:
                if mask is not None:
                    if not isinstance(_scale, list):
                        _scale = [_scale] * mask.shape[1]

                    current_num_images = mask.shape[1]
                    for i in range(current_num_images):
                        ip_key = to_k_ip(current_ip_hidden_states[:, i, :, :])
                        ip_value = to_v_ip(current_ip_hidden_states[:, i, :, :])

                        ip_key = ip_key.view(
                            batch_size, -1, attn.heads, head_dim
                        ).transpose(1, 2)
                        ip_value = ip_value.view(
                            batch_size, -1, attn.heads, head_dim
                        ).transpose(1, 2)

                        # the output of sdp = (batch, num_heads, seq_len, head_dim)
                        # TODO: add support for attn.scale when we move to Torch 2.1
                        _current_ip_hidden_states = F.scaled_dot_product_attention(
                            query,
                            ip_key,
                            ip_value,
                            attn_mask=None,
                            dropout_p=0.0,
                            is_causal=False,
                        )

                        _current_ip_hidden_states = _current_ip_hidden_states.transpose(
                            1, 2
                        ).reshape(batch_size, -1, attn.heads * head_dim)
                        _current_ip_hidden_states = _current_ip_hidden_states.to(
                            query.dtype
                        )

                        mask_downsample = IPAdapterMaskProcessor.downsample(
                            mask[:, i, :, :],
                            batch_size,
                            _current_ip_hidden_states.shape[1],
                            _current_ip_hidden_states.shape[2],
                        )

                        mask_downsample = mask_downsample.to(
                            dtype=query.dtype, device=query.device
                        )
                        hidden_states = hidden_states + _scale[i] * (
                            _current_ip_hidden_states * mask_downsample
                        )
                else:
                    ip_key = to_k_ip(current_ip_hidden_states)
                    ip_value = to_v_ip(current_ip_hidden_states)

                    ip_key = ip_key.view(
                        batch_size, -1, attn.heads, head_dim
                    ).transpose(1, 2)
                    ip_value = ip_value.view(
                        batch_size, -1, attn.heads, head_dim
                    ).transpose(1, 2)

                    # the output of sdp = (batch, num_heads, seq_len, head_dim)
                    # TODO: add support for attn.scale when we move to Torch 2.1
                    current_ip_hidden_states = F.scaled_dot_product_attention(
                        query,
                        ip_key,
                        ip_value,
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=False,
                    )

                    current_ip_hidden_states = current_ip_hidden_states.transpose(
                        1, 2
                    ).reshape(batch_size, -1, attn.heads * head_dim)
                    current_ip_hidden_states = current_ip_hidden_states.to(query.dtype)

                    hidden_states = hidden_states + _scale * current_ip_hidden_states
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if (cross_attn and ca_cache) or (self_attn and sa_cache):
            if keep_shape:
                cache = hidden_states
            else:
                hidden_uncond, hidden_pred_text = hidden_states.chunk(2)
                cache = (hidden_uncond + hidden_pred_text ) / 2
        else:
            cache = None

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    return hidden_states, cache





def tgate_scheduler(
    cur_step: int = None, 
    gate_step: int = 10, 
    sp_interval: int = 5,
    fi_interval: int = 1,
    warm_up: int = 2,
):
    r"""
    The T-GATE scheduler function 

    Args:
        cur_step (`int`):
            The current time step. 
        gate_step (`int` defaults to 10): 
            The time step to stop calculating the cross attention.
        sp_interval (`int` defaults to 5): 
            The time-step interval to cache self attention before gate_step (Semantics-Planning Phase).
        fi_interval (`int` defaults to 1): 
            The time-step interval to cache self attention after gate_step (Fidelity-Improving Phase).
        warm_up (`int` defaults to 2): 
            The time step to warm up the model inference.

    Returns:
        ca_kward: (`dict`):
            A kwargs dictionary to pass along to the cross attention for caching and reusing. 
        sa_kward: (`dict`):
            A kwargs dictionary to pass along to the self attention for caching and reusing. 
        keep_shape (`bool`):
            Whether or not to remain the shape of hidden features
    """
    if cur_step < gate_step-1:
        # Semantics-Planning Stage
        ca_kwards = {
            'cache': False,
            'reuse': False,
        }
        if cur_step < warm_up:
            sa_kwards = {
                'cache': False,
                'reuse': False,
            }
        elif cur_step == warm_up:
            sa_kwards = {
                'cache': True,
                'reuse': False,
            }   
        else:
            if cur_step % sp_interval == 0:
                sa_kwards = {
                    'cache': True,
                    'reuse': False,
                }
            else:
                sa_kwards = {
                    'cache': False,
                    'reuse': True,
                }
        keep_shape = True
    
    elif cur_step == gate_step-1:
        ca_kwards = {
            'cache': True,
            'reuse': False,
        }
        sa_kwards = {
            'cache':True,
            'reuse':False
        }
        keep_shape = False
    else:
        # Fidelity-Improving Stage
        ca_kwards = {
            'cache': False,
            'reuse': True,
        }
        if cur_step % fi_interval == 0:
            sa_kwards = {
                'cache':True,
                'reuse':False
            }
        else:
            sa_kwards = {
                'cache':False,
                'reuse':True
            }
        keep_shape = True
    return ca_kwards, sa_kwards, keep_shape