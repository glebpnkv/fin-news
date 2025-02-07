import torch
from transformers import InformerModel, InformerConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqTSModelOutput
from transformers.models.informer.modeling_informer import (
    InformerEncoder, 
    InformerDecoder, 
    InformerFeatureEmbedder, 
    InformerNOPScaler, 
    InformerMeanScaler, 
    InformerStdScaler
)
from torch import nn


class InformerEncoderFusion(InformerEncoder):
    """
    Informer encoder which allows additional inputs (e.g. embeddings of news articles).
    """
    def __init__(
        self,
        config: InformerConfig
    ):
        super().__init__(config)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        inputs_fusion: torch.FloatTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | BaseModelOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.value_embedding(inputs_embeds)
        embed_pos = self.embed_positions(inputs_embeds.size())
        
        # Adding fusion inputs if they are provided
        if inputs_fusion is not None:
            hidden_states = hidden_states + inputs_fusion
        
        hidden_states = self.layernorm_embedding(hidden_states + embed_pos)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        for idx, (encoder_layer, conv_layer) in enumerate(zip(self.layers, self.conv_layers)):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                    if conv_layer is not None:
                        output = self._gradient_checkpointing_func(conv_layer, layer_outputs[0])
                        layer_outputs = (output,) + layer_outputs[1:]
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )
                    if conv_layer is not None:
                        output = conv_layer(layer_outputs[0])
                        layer_outputs = (output,) + layer_outputs[1:]

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class InformerFusionModel(InformerModel):
    def __init__(
        self,
        config: InformerConfig
    ):
        super(InformerModel, self).__init__(config)
        
        if config.scaling == "mean" or config.scaling is True:
            self.scaler = InformerMeanScaler(config)
        elif config.scaling == "std":
            self.scaler = InformerStdScaler(config)
        else:
            self.scaler = InformerNOPScaler(config)

        if config.num_static_categorical_features > 0:
            self.embedder = InformerFeatureEmbedder(
                cardinalities=config.cardinality,
                embedding_dims=config.embedding_dimension,
            )

        # transformer encoder-decoder and mask initializer
        self.encoder = InformerEncoderFusion(config)
        self.decoder = InformerDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()

        # self.output = TODO
        
    def forward(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        past_observed_mask: torch.Tensor,
        static_categorical_features: torch.Tensor | None = None,
        static_real_features: torch.Tensor | None = None,
        future_values: torch.Tensor | None = None,
        future_time_features: torch.Tensor | None = None,
        inputs_fusion: torch.FloatTensor | None = None,
        decoder_attention_mask: torch.LongTensor | None = None,
        head_mask: torch.Tensor | None = None,
        decoder_head_mask: torch.Tensor | None = None,
        cross_attn_head_mask: torch.Tensor | None = None,
        encoder_outputs: list[torch.FloatTensor] | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
    ) -> Seq2SeqTSModelOutput | tuple:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_inputs, loc, scale, static_feat = self.create_network_inputs(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
            static_categorical_features=static_categorical_features,
            static_real_features=static_real_features,
            future_values=future_values,
            future_time_features=future_time_features,
        )

        if encoder_outputs is None:
            enc_input = transformer_inputs[:, : self.config.context_length, ...]
            encoder_outputs = self.encoder(
                inputs_embeds=enc_input,
                inputs_fusion=inputs_fusion,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        dec_input = transformer_inputs[:, self.config.context_length :, ...]
        decoder_outputs = self.decoder(
            inputs_embeds=dec_input,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs + (loc, scale, static_feat)

        return Seq2SeqTSModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            loc=loc,
            scale=scale,
            static_features=static_feat,
        )
