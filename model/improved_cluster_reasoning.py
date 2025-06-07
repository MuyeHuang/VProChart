import numpy as np
from typing import Dict, Optional, Tuple, Union, Any
import os
import json
from random import shuffle
import time
import sys
import logging
import transformers
from transformers import PreTrainedModel, ViTModel, ViTFeatureExtractor, LxmertXLayer, TapasModel, TapasForSequenceClassification,BertTokenizer, BertModel,BertLayer
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.pix2struct.modeling_pix2struct import Pix2StructVisionConfig,Pix2StructVisionEncoder,Pix2StructVisionMlp,Pix2StructVisionAttention,Pix2StructVisionLayer,Pix2StructTextModel,Pix2StructVisionEmbeddings,Pix2StructLayerNorm,BaseModelOutputWithPooling,Union,BaseModelOutput,Seq2SeqModelOutput,Seq2SeqLMOutput
from transformers import TapasConfig, LxmertConfig, ViTConfig, ViTPreTrainedModel, PreTrainedModel, BertConfig, Pix2StructVisionModel, Pix2StructForConditionalGeneration ,Pix2StructConfig ,Pix2StructVisionConfig
from transformers.models.vit.modeling_vit import ViTEncoder
from transformers.models.bert.modeling_bert import BertLayer,BertConfig,BertModel
from transformers import LxmertConfig, LxmertXLayer
import torch.nn as nn
from transformers import TapasTokenizer
import torch
import pandas as pd
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
from transformers import AdamW
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import EvalPrediction
from transformers import Trainer, TrainingArguments
import collections
from transformers.activations import gelu
# from .config import VisionTapasConfig
from dataclasses import dataclass
from sklearn.cluster import AgglomerativeClustering
from transformers import BertConfig,BertLayer,XLMRobertaConfig,XLMRobertaModel
class GeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


class ClusterReasoning(nn.Module):
    def __init__(self,):
        super().__init__()
        self.xlm_config_for_query = XLMRobertaConfig(num_hidden_layers=1,
                                                     vocab_size=57531)
        self.query_reasoning = XLMRobertaModel(self.xlm_config_for_query)
        self.text_project_layer = nn.Linear(768,1024)
        self.xlm_config_for_visual = XLMRobertaConfig(num_hidden_layers=1,
                                                      max_position_embeddings=1024,
                                                      hidden_size=1024,
                                                      num_attention_heads=16,
                                                      max_length=1024)
        self.cluster_base_reasoning = XLMRobertaModel(self.xlm_config_for_visual)
    def forward(self ,vision_state ,alignment_state ,textual_ids ,cluster_state):
        batch_size, _ = textual_ids.size()
        # indices = torch.nonzero(textual_ids == 57530, as_tuple=False)
        # insize, _ =indices.size()
        # if insize == batch_size:
        #     for i in range(batch_size):
        #         index = indices[i, 1].item()
        #         textual_ids[i, index:] = 1
        text_mask = textual_ids.clone()
        text_mask[textual_ids == 1] = 0
        text_mask[textual_ids != 1] = 1
        text_output = self.query_reasoning(input_ids=textual_ids,
                                           attention_mask=text_mask)
        text_feat = self.text_project_layer(text_output.pooler_output)
        cluster_mask = cluster_state.clone()
        
        for j in range(5):
            cluster_mask[cluster_state == j] = 1
            cluster_mask[cluster_state != j] = 0
            output_state = self.cluster_base_reasoning(inputs_embeds=vision_state,
                                                       attention_mask=cluster_mask)
            vision_state = output_state.last_hidden_state
        
        
        patch_score = torch.matmul(vision_state,text_feat.unsqueeze(-1))
        
        patch_prob = torch.nn.functional.softmax(patch_score.squeeze(2),dim=1)
        
        _, k_ind = torch.topk(patch_prob,5,dim=1) 
        for k in range(batch_size):
            ind = torch.zeros((batch_size,900,900),device=vision_state.device)
            ind[k,k_ind[k,:],:] = 1
            alignment_state[ind == 0] = 0.
        return vision_state,alignment_state


def index_to_position(index, col):
    row_p = index//col + 1
    col_p = index%col + 1
    return torch.cat((row_p.unsqueeze(0),col_p.unsqueeze(0)))

class ClusterAlignmentUnit(nn.Module):
    def __init__(self,):
        super().__init__()
        self.align_cls = nn.Sequential(nn.GELU(),nn.Linear(1024,64),
                                       nn.GELU(),nn.Linear(64,1),nn.Softmax(dim=1))
        # self.align_weight = nn.Softmax(dim=1)
        self.transpose_to_semantic = nn.Sequential(nn.GELU(), nn.Linear(1024,1024))
        self.get_aligned = nn.Softmax(dim=1)
        self.cluster_pooler = ClusterFeatureExtractor()
        self.align_matrix_to_token = nn.Sequential(nn.GELU(), nn.Linear(900,1024))
        self.global_to_weight = nn.Sequential(nn.GELU(),nn.Linear(1024,3),nn.Softmax(dim=0))
        self.layer_norm = nn.LayerNorm(normalized_shape=1024)
        near_align_matrix = torch.zeros((900, 900))
        patch_ind = torch.arange(0,900)
        for ind in patch_ind:
            current_posi = index_to_position(ind, torch.tensor(30))
            for ind2 in patch_ind:
                if ind == ind2:
                    continue
                comp_posi = index_to_position(ind2, torch.tensor(30))
                distance = torch.sqrt(torch.sum((current_posi - comp_posi) ** 2))
                if distance <= 3:
                    near_align_matrix[ind,ind2] = 1.
        cross_align_matrix = torch.zeros((900, 900))
        for ind in patch_ind:
            current_posi = index_to_position(ind, torch.tensor(30))
            for ind2 in patch_ind:
                if ind == ind2:
                    continue
                comp_posi = index_to_position(ind2, torch.tensor(30))
                if comp_posi[0]-current_posi[0] == 0 or comp_posi[1]-current_posi[1] == 0:
                    cross_align_matrix[ind,ind2] = 1.
        self.near_matrix = near_align_matrix
        self.cross_matrix = cross_align_matrix
        self.cluster_reasoning = ClusterReasoning()

    def forward(self ,vit_sequence, cluster_patch, coordinate_info, pooled_feat, text_ids):

        device = vit_sequence.device
        batch_size, num_samples, feature_dim = vit_sequence.size()
        cluster, cluster_mask, cluster_pooling_output = self.cluster_pooler(vit_sequence, cluster_patch)
        pooled_cluster_weight = self.align_cls(cluster_pooling_output).squeeze()
        if pooled_cluster_weight.dim() == 1:
            pooled_cluster_weight = pooled_cluster_weight.unsqueeze(0)

        aliment_info = []
        alignment_token_list = []
        alignment_matrix_output = []
        
        cluster_num = torch.max(cluster_patch).item()+1
        
        align_mask_list = []
        cluster_patch_num = (cluster_patch+1).float()
        align_mask = torch.bmm(cluster_patch_num.unsqueeze(2), cluster_patch_num.unsqueeze(1))
        diag_mask = torch.eye(900).unsqueeze(0).expand(batch_size, -1, -1).to(device)
        align_mask = align_mask*(1.0 - diag_mask)
        compare_ind = [2.,3.,4.,5.,6.,8.,10.,12.,15.,20.]
        cluster_weight_ind_matrix = torch.bmm(pooled_cluster_weight.unsqueeze(2), pooled_cluster_weight.unsqueeze(1))
        cluster_weight_ind = [cluster_weight_ind_matrix[:,0,1],cluster_weight_ind_matrix[:,0,2],cluster_weight_ind_matrix[:,0,3],cluster_weight_ind_matrix[:,0,4],
                              cluster_weight_ind_matrix[:,1,2],cluster_weight_ind_matrix[:,1,3],cluster_weight_ind_matrix[:,1,4],
                              cluster_weight_ind_matrix[:,2,3],cluster_weight_ind_matrix[:,2,4],
                              cluster_weight_ind_matrix[:,3,4]]
        cluster_weight = torch.ones_like(align_mask, dtype=vit_sequence.dtype).to(device)
        for ind in range(len(compare_ind)):
            align_mask_ind = align_mask.clone()
            cluster_weight_num = cluster_weight_ind[ind].unsqueeze(1).unsqueeze(1).expand(batch_size, 900, 900)
            cluster_weight[align_mask == compare_ind[ind]] = cluster_weight_num[align_mask == compare_ind[ind]]
            align_mask_ind[align_mask == compare_ind[ind]] = 0
            align_mask_ind[align_mask != compare_ind[ind]] = -60000.
            align_mask_list.append(align_mask_ind)
        align_mask_stack = torch.stack(align_mask_list,dim=1)
        align_mask_list.clear()
        aliment_weight = self.global_to_weight(pooled_feat)

        
        semantic_state = self.transpose_to_semantic(vit_sequence)
        semantic_alignment = torch.bmm(semantic_state, semantic_state.transpose(1, 2))
        semantic_alignment_stack = semantic_alignment.unsqueeze(1).repeat_interleave(len(compare_ind), dim=1)
        semantic_alignment_softmaxed = torch.nn.functional.softmax(semantic_alignment_stack+align_mask_stack, dim=3)
        semantic_alignment_softmaxedk = semantic_alignment_softmaxed.clone()
        semantic_alignment_softmaxedk[align_mask_stack == -60000.] = 0.
        semantic_alignment_state = torch.sum(semantic_alignment_softmaxedk, dim=1)
        
        align_state = semantic_alignment_state.clone()
        for i in range(batch_size):
            align_state[i] = semantic_alignment_state[i]*aliment_weight[i,0]+self.cross_matrix.to(device)*aliment_weight[i,1]+self.near_matrix.to(device)*aliment_weight[i,2]
        
        reasoned_vit_sequence, align_ind = self.cluster_reasoning(vit_sequence , align_state*cluster_weight,text_ids , cluster_patch)
        alignment_token = self.align_matrix_to_token(align_ind)
        aligned_vit_sequence = self.layer_norm(vit_sequence + alignment_token + reasoned_vit_sequence)
      
        return aligned_vit_sequence, aliment_info, cluster_pooling_output, (cluster, cluster_mask, align_state, cluster_patch)

class ClusterFeatureExtractor(nn.Module):
    def __init__(self,):
        super().__init__()
        self.bert_config = BertConfig(hidden_size=1024,num_attention_heads=16)
        self.vision_pooler = BertLayer(self.bert_config)

    def forward(self ,vision_state, cluster_state):
        device = vision_state.device
        batch_size, num_samples, feature_dim = vision_state.size()
        cluster = [torch.zeros(batch_size, 1000, 1024).to(device=device) for _ in range(5)]
        cluster_mask = [torch.zeros(batch_size, 1000).to(device=device) for _ in range(5)]
        cluster_pooling_output = []
        for i in range(5):
            for j in range(batch_size):
                incident = (cluster_state[j,:] == i).unsqueeze(1).expand(-1,1024)
                cluster_token = vision_state[j,:,:][incident].view(-1,1024)
                # print(j+1,i+1,cluster_token.shape[0])
                if cluster_token.shape[0] <=1000:
                    cluster[i][j,:cluster_token.shape[0],:] = cluster_token
                else:
                    cluster[i][j, :cluster_token.shape[0], :] = cluster_token[:1000,:]
                cluster_token_mask = torch.zeros(1000,)
                cluster_token_mask[:torch.sum(cluster_state[j,:] == i).item()] = 1
                cluster_mask[i][j,:] = cluster_token_mask
            pooling_padding = torch.zeros(batch_size,1,1024).to(device=device)
            cluster_pooling_state = self.vision_pooler(torch.cat((pooling_padding,cluster[i][:,:511]),dim=1),
                                                       cluster_mask[i][:,:512].unsqueeze(1).unsqueeze(1))
            cluster_pooling_output.append(cluster_pooling_state[0][:,:1,:])
        return (cluster, cluster_mask, torch.cat(cluster_pooling_output, dim=1))

class HierarchicalClusteringLayer(nn.Module):
    def __init__(self, input_dim, num_clusters):
        super(HierarchicalClusteringLayer, self).__init__()
        self.input_dim = input_dim
        self.num_clusters = num_clusters
        self.clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
        self.mlp_layer = nn.Sequential(nn.GELU(),
                                       nn.Linear(1024,256),
                                       nn.GELU(),
                                       nn.Linear(256,32))
    def forward(self, x):

        x = self.mlp_layer(x)
        batch_size, num_samples, feature_dim = x.size()

        x_reshaped = x.view(-1, feature_dim)

        features_np = x_reshaped.cpu().detach().numpy()

        labels = []
        for i in range(batch_size):
            clustering = AgglomerativeClustering(n_clusters=self.num_clusters, linkage='ward')
            batch_labels = clustering.fit_predict(features_np[i * num_samples: (i + 1) * num_samples])
            labels.append(batch_labels)

        labels_tensor = torch.from_numpy(np.concatenate(labels)).view(batch_size, num_samples, -1).to(x.device)

        return labels_tensor.squeeze()


class MultiModalEnhanceAttention(Pix2StructVisionAttention):
    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_attention_heads
        self.dropout = config.attention_dropout
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.query = nn.Linear(self.hidden_size, self.inner_dim, bias=False)
        self.key = nn.Linear(self.hidden_size, self.inner_dim, bias=False)
        self.value = nn.Linear(self.hidden_size, self.inner_dim, bias=False)
        self.output = nn.Linear(self.inner_dim, self.hidden_size, bias=False)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        output_attentions=False,
    ):
        """
        Self-attention block
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        def to_projection_shape(states):
            """projection"""
            return states.contiguous().view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        # get query states
        # (batch_size, n_heads, seq_length, dim_per_head)
        query_states = to_projection_shape(self.query(hidden_states))

        # get key/value states
        key_states = to_projection_shape(self.key(hidden_states))
        value_states = to_projection_shape(self.value(hidden_states))

        # compute scores
        # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        position_bias_mask = torch.zeros(
            (1, self.n_heads, seq_length, seq_length), device=scores.device, dtype=scores.dtype
        )

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=scores.device, dtype=scores.dtype)

        if attention_mask.dim() == 2:
            position_bias_mask = position_bias_mask + (attention_mask[:, None, None, :]-1).to(position_bias.device)
        else:
            # (batch_size, n_heads, seq_length, key_length)
            position_bias_mask = position_bias_mask + (attention_mask-1).to(position_bias.device)
        position_bias_mask = position_bias_mask.masked_fill(position_bias_mask == -1, torch.finfo(scores.dtype).min)

        position_bias_masked = position_bias.masked_fill(position_bias == 1, torch.finfo(scores.dtype).min)+position_bias_mask
        scores += position_bias_masked
        scores = torch.max(scores, torch.tensor(torch.finfo(scores.dtype).min))

        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).type_as(scores)

        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = torch.matmul(attn_weights, value_states)

        # (batch_size, seq_length, dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        attn_output = self.output(attn_output)

        outputs = (attn_output,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class MultimodalLayer(Pix2StructVisionLayer):
    def __init__(self, config: Pix2StructConfig) -> None:
        super().__init__(config)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = MultiModalEnhanceAttention(config)
        self.mlp = Pix2StructVisionMlp(config)
        self.pre_mlp_layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pre_attention_layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.get_q = Pix2StructVisionMlp(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        text_states: Optional[torch.FloatTensor] = None,
        text_attention_mask: Optional[torch.FloatTensor] = None,
        enhance_level: Optional[int] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        residual = hidden_states

        # in Pix2StructVision, layernorm is applied before self-attention
        hidden_states = self.pre_attention_layer_norm(hidden_states)

        enhance_ratio = 2
        query = self.get_q(text_states[:,0,:]).unsqueeze(2)
        attention_enhance_array = torch.matmul(hidden_states, query).transpose(1, 2)
        sorted_indices = torch.argsort(attention_enhance_array,dim=2)

        attention_enhance_array[:,:,sorted_indices[:,:,2048//((enhance_level+2)*2):]] = 0.
        attention_enhance_score = 1 + (torch.matmul(attention_enhance_array.transpose(1, 2),attention_enhance_array))*enhance_ratio
        attention_enhance_score = torch.max(attention_enhance_score, torch.tensor(torch.finfo(attention_enhance_score.dtype).min)).unsqueeze(1)
        # attention_enhance_weights = nn.functional.softmax(attention_enhance_score, dim=-1, dtype=torch.float32).type_as(attention_enhance_score)
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=head_mask,
            output_attentions=output_attentions,
            position_bias=attention_enhance_score,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + residual

        # in Pix2StructVision, layernorm is also applied after self-attention
        layer_output = self.pre_mlp_layer_norm(hidden_states)
        layer_output = self.mlp(layer_output) + hidden_states  # second residual connection

        outputs = (layer_output,) + outputs

        return outputs


class MultimodalEncoder(Pix2StructVisionEncoder):
    def __init__(self, config: Pix2StructConfig) -> None:
        super().__init__(config)
        self.config = config
        self.layer = nn.ModuleList([Pix2StructVisionLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        self.c_layer = nn.ModuleList([MultimodalLayer(config) for _ in range(4)])
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        text_states: Optional[torch.FloatTensor] = None,
        text_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        for i, layer_module in enumerate(self.c_layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions, text_states, text_attention_mask, i)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)


        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class MultimodalModel(Pix2StructVisionModel):
    config_class = Pix2StructVisionConfig
    main_input_name = "flattened_patches"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Pix2StructVisionLayer"]

    def __init__(self, config: Pix2StructConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = Pix2StructVisionEmbeddings(config)
        self.encoder = MultimodalEncoder(config)

        self.layernorm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        flattened_patches: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        text_states: Optional[torch.FloatTensor] = None,
        text_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> BaseModelOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if flattened_patches is None:
            raise ValueError("You have to specify flattened_patches")

        if attention_mask is None:
            # check where `flattened_patches` is not 0
            attention_mask = (flattened_patches.sum(dim=-1) != 0).float()

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(flattened_patches)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            text_states=text_states,
            text_attention_mask=text_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            head_outputs = (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class MultiModalPix2Struct(Pix2StructForConditionalGeneration):
    config_class = Pix2StructConfig
    main_input_name = "flattened_patches"
    _tied_weights_keys = ["decoder.lm_head.weight"]

    def __init__(self, config: Pix2StructConfig):
        super().__init__(config)

        self.encoder = MultimodalModel(config.vision_config)
        self.decoder = Pix2StructTextModel(config.text_config)

        self.is_vqa = config.is_vqa

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        flattened_patches: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        text_states: Optional[torch.FloatTensor] = None,
        text_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Seq2SeqLMOutput:
        use_cache = use_cache if use_cache is not None else self.config.text_config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                flattened_patches=flattened_patches,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                text_states=text_states,
                text_attention_mask=text_attention_mask,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)
            decoder_attention_mask = (
                decoder_attention_mask
                if decoder_attention_mask is not None
                else decoder_input_ids.ne(self.config.pad_token_id).float()
            )
            # Always attend to the first token
            decoder_attention_mask[:, 0] = 1

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            labels=labels,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=decoder_outputs.loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self,
            input_ids,
            flattened_patches: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            past_key_values=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs,
    ):
        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones_like(input_ids).to(input_ids.device)

        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "flattened_patches": flattened_patches,
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }
