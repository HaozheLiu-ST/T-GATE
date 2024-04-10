import torch
import torch.nn.functional as F
from diffusers.utils import (
    BACKENDS_MAPPING,
    deprecate,
    logging
)
import inspect
from typing import Callable, List, Optional, Tuple, Union
logger = logging.get_logger(__name__) 

def register_tgate_forward(
    model, 
    filter_name: str = 'Attention', 
    gate_step: int = 10,
    inference_num_per_image: int = 25,
    lcm: bool = False,
    cur_step: int = None,
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
        gate_step (`int`):
            The time step to stop calculating the cross attention.
        inference_num_per_image (`int`):
            The total inference steps for generating one image.
        lcm (`bool`):
            whether or not the latent consistency model is used.

    Returns:
        count (`int`): The number of the cross attention layers used in the given model.
    """

    count = 0
    def warp_custom(
        self: torch.nn.Module,
        gate_step: int = None,
        inference_num_per_image: int = None,
        lcm: bool = False,
        cur_step: int = None,
        **kwargs,
    ):          
        def forward(
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            gate_step = gate_step,
            inference_num_per_image = inference_num_per_image,
            lcm = lcm,
            cur_step = cur_step,
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
                gate_step (`int`):
                    The time step to stop calculating the cross attention.
                inference_num_per_image (`int`):
                    The total inference steps for generating one image.
                lcm (`bool`):
                    whether or not the latent consistency model is used.
                **cross_attention_kwargs:
                    Additional keyword arguments to pass along to the cross attention.

            Returns:
                `torch.Tensor`: The output of the attention layer.
            """
            # The `Attention` class can call different attention processors / attention functions
            # here we simply pass along all tensors to the selected processor class
            # For standard processors that are defined here, `**cross_attention_kwargs` is empty
            if not hasattr(self,'cur_step'):
                self.cur_step = 1
            else:
                if cur_step != None:
                    self.cur_step = cur_step
                else:
                    self.cur_step =   (self.cur_step + 1) % inference_num_per_image


            if not hasattr(self,'cache'):
                self.cache = None
            attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
            unused_kwargs = [k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters]
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
                gate_step=gate_step,
                cur_step = self.cur_step,
                cache = self.cache,
                lcm = lcm,
                **cross_attention_kwargs,
            )
            if gate_step == self.cur_step:
                self.cache = cache
            return hidden_states
        return forward

    def register_recr(
        net: torch.nn.Module, 
        count: int = None, 
        gate_step: int = None, 
        inference_num_per_image: int = None,
        lcm: bool = False,
        cur_step:int = None
    ):
        if net.__class__.__name__ == filter_name:
            net.forward = warp_custom(net, gate_step=gate_step, inference_num_per_image=inference_num_per_image, lcm=lcm, cur_step=cur_step)
            return count + 1
        elif hasattr(net, 'children'):
            for net_child in net.children():
                count = register_recr(net_child, count, gate_step=gate_step, inference_num_per_image=inference_num_per_image, lcm=lcm,cur_step=cur_step)
        return count

    return register_recr(model, count, gate_step=gate_step, inference_num_per_image=inference_num_per_image, lcm=lcm,cur_step=cur_step) 




def tgate_processor(
    attn =None,
    hidden_states = None,
    encoder_hidden_states = None,
    attention_mask  = None,
    temb  = None,
    gate_step=None,
    cur_step = None,
    cache = None,
    lcm = False,
    *args,
    **kwargs,
) -> torch.FloatTensor:

    if not hasattr(F, "scaled_dot_product_attention"):
        raise ImportError(
            "AttnAddedKVProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
        )
    if len(args) > 0 or kwargs.get("scale", None) is not None:
        deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
        deprecate("scale", "1.0.0", deprecation_message)
    residual = hidden_states

    cross_attn = encoder_hidden_states is not None

    input_ndim = hidden_states.ndim

    # if the attention is cross-attention or self-attention
    if cross_attn and (gate_step<cur_step):
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

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)


        if cross_attn and (gate_step == cur_step):
            if lcm:
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

