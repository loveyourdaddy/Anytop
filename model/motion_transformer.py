import torch 
import torch.nn as nn
from typing import Optional, Union, Callable, Tuple
from torch import Tensor
import torch.nn.functional as F
from torch.nn import MultiheadAttention
CUDA_LAUNCH_BLOCKING=1

class GraphMultiHeadAttention(nn.Module):
    def __init__(self, d_model, dropout, nheads):
        super().__init__()

        self.nheads = nheads

        self.att_size = att_size = d_model // nheads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(d_model, nheads * att_size)
        self.linear_k = nn.Linear(d_model, nheads * att_size)
        self.linear_v = nn.Linear(d_model, nheads * att_size)
        self.dropout = nn.Dropout(dropout)

        self.output_layer = nn.Linear(nheads * att_size, d_model)

    def forward(
        self,
        q,
        k,
        v,
        query_hop_emb,
        query_edge_emb,
        key_hop_emb,
        key_edge_emb,
        value_hop_emb,
        value_edge_emb,
        distance,
        edge_attr,
        mask=None,
    ):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.nheads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.nheads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.nheads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2)  # [b, h, k_len, d_k]

        sequence_length = v.shape[2]
        num_hop_types = query_hop_emb.shape[0]
        num_edge_types = query_edge_emb.shape[0]

        query_hop_emb = query_hop_emb.view(
            1, num_hop_types, self.nheads, self.att_size
        ).transpose(1, 2)
        query_edge_emb = query_edge_emb.view(
            1, -1, self.nheads, self.att_size
        ).transpose(1, 2)
        key_hop_emb = key_hop_emb.view(
            1, num_hop_types, self.nheads, self.att_size
        ).transpose(1, 2)
        key_edge_emb = key_edge_emb.view(
            1, num_edge_types, self.nheads, self.att_size
        ).transpose(1, 2)

        query_hop = torch.matmul(q, query_hop_emb.transpose(2, 3))
        query_hop = torch.gather(
            query_hop, 3, distance.unsqueeze(1).repeat(1, self.nheads, 1, 1)
        )
        query_edge = torch.matmul(q, query_edge_emb.transpose(2, 3))
        query_edge = torch.gather(
            query_edge, 3, edge_attr.unsqueeze(1).repeat(1, self.nheads, 1, 1)
        )

        key_hop = torch.matmul(k, key_hop_emb.transpose(2, 3))
        key_hop = torch.gather(
            key_hop, 3, distance.unsqueeze(1).repeat(1, self.nheads, 1, 1)
        )
        key_edge = torch.matmul(k, key_edge_emb.transpose(2, 3))
        key_edge = torch.gather(
            key_edge, 3, edge_attr.unsqueeze(1).repeat(1, self.nheads, 1, 1)
        )

        spatial_bias = (query_hop + key_hop)
        edge_bais = (query_edge + key_edge)

        x = torch.matmul(q, k.transpose(2, 3)) + spatial_bias + edge_bais

        x = x * self.scale

        if mask is not None:
            x = x + mask

        x = torch.softmax(x, dim=3)
        x = self.dropout(x)
        if value_hop_emb is not None:
            value_hop_emb = value_hop_emb.view(
                1, num_hop_types, self.nheads, self.att_size
            ).transpose(1, 2)
            value_edge_emb = value_edge_emb.view(
                1, num_edge_types, self.nheads, self.att_size
            ).transpose(1, 2)

            value_hop_att = torch.zeros(
                (batch_size, self.nheads, sequence_length, num_hop_types),
                device=value_hop_emb.device,
            )
            value_hop_att = torch.scatter_add(
                value_hop_att, 3, distance.unsqueeze(1).repeat(1, self.nheads, 1, 1), x
            )
            value_edge_att = torch.zeros(
                (batch_size, self.nheads, sequence_length, num_edge_types),
                device=value_hop_emb.device,
            )
            value_edge_att = torch.scatter_add(
                value_edge_att, 3, edge_attr.unsqueeze(1).repeat(1, self.nheads, 1, 1), x
            )
        x = torch.matmul(x, v)
        if value_hop_emb is not None:
            x = x + torch.matmul(value_hop_att, value_hop_emb) + torch.matmul(value_edge_att, value_edge_emb)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.nheads * d_v)

        x = self.output_layer(x)
        assert x.size() == orig_q_size
        return x

class TposeCrossAttention(nn.Module):
    """
    Target T-pose와 source T-pose 사이의 joint 대응 관계를 학습한다.
    출력 bias [bs, nheads, tgt_joints, src_joints]는 cross-motion attention의 prior로 사용된다.
    """
    def __init__(self, d_model, nheads):
        super().__init__()
        self.nheads = nheads
        self.att_size = d_model // nheads
        self.scale = self.att_size ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)

    def forward(self, tgt_tpose, src_tpose):
        """
        Args:
            tgt_tpose: [bs, tgt_joints, d_model]
            src_tpose: [bs, src_joints, d_model]
        Returns:
            bias: [bs, nheads, tgt_joints, src_joints]
        """
        bs, tgt_j, _ = tgt_tpose.shape
        src_j = src_tpose.shape[1]
        q = self.q_proj(tgt_tpose).view(bs, tgt_j, self.nheads, self.att_size).transpose(1, 2)
        k = self.k_proj(src_tpose).view(bs, src_j, self.nheads, self.att_size).transpose(1, 2)
        return torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [bs, nheads, tgt_j, src_j]


class CrossMotionAttention(nn.Module):
    """
    Target motion features(Q)가 source motion features(K, V)를 attend한다.
    TposeCrossAttention으로 계산된 bias를 통해 T-pose 기반 joint 대응 관계가 반영된다.
    """
    def __init__(self, d_model, nheads, dropout=0.1):
        super().__init__()
        self.nheads = nheads
        self.att_size = d_model // nheads
        self.scale = self.att_size ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, src, tpose_bias=None, src_mask=None):
        """
        Args:
            tgt:        [B, tgt_j, d]
            src:        [B, src_j, d]
            tpose_bias: [B, nheads, tgt_j, src_j]  T-pose 대응 bias
            src_mask:   [B, 1, 1, src_j]           padding 마스크 (-1e9)
        Returns:
            [B, tgt_j, d]
        """
        B, tgt_j, d = tgt.shape
        src_j = src.shape[1]
        q = self.q_proj(tgt).view(B, tgt_j, self.nheads, self.att_size).transpose(1, 2)
        k = self.k_proj(src).view(B, src_j, self.nheads, self.att_size).transpose(1, 2)
        v = self.v_proj(src).view(B, src_j, self.nheads, self.att_size).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, nheads, tgt_j, src_j]
        if tpose_bias is not None:
            scores = scores + tpose_bias
        if src_mask is not None:
            scores = scores + src_mask
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, tgt_j, d)
        return self.out_proj(out)


class GraphMotionDecoder(nn.TransformerDecoder):
    def __init__(self, decoder_layer, num_layers, norm=None, max_path_len=5, value_emb=False): 
                # multi head attention
        super().__init__(decoder_layer, num_layers, norm)
        
        self.d_model = decoder_layer.d_model
        self.topology_key_emb = nn.Embedding(max_path_len + 1, self.d_model) # 'far': max_path_len + 1
        self.edge_key_emb = nn.Embedding(6, self.d_model) # 'self':0, 'parent':1, 'child':2, 'sibling':3, 'no_relation':4, 'end_effector':5
        self.topology_query_emb = nn.Embedding(max_path_len + 1, self.d_model) # 'far': max_path_len + 1
        self.edge_query_emb = nn.Embedding(6, self.d_model) # 'self':0, 'parent':1, 'child':2, 'sibling':3, 'no_relation':4, 'end_effector':5
        self.value_emb_flag = value_emb
        if value_emb:
            self.topology_value_emb = nn.Embedding(max_path_len + 1, self.d_model) # 'far': max_path_len + 1
            self.edge_value_emb = nn.Embedding(6, self.d_model) # 'self':0, 'parent':1, 'child':2, 'sibling':3, 'no_relation':4, 'end_effector':5
        

        
    def forward(self, tgt: Tensor, timesteps_embs: Tensor, memory: Tensor, spatial_mask:  Optional[Tensor] = None,
                temporal_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, y=None, get_layer_activation=-1,
                source_memory: Optional[Tensor] = None, tpose_bias: Optional[Tensor] = None,
                src_cross_mask: Optional[Tensor] = None) -> Union[Tensor , Tuple[Tensor, dict]]:
        topology_rel = y['graph_dist'].long().to(tgt.device)
        edge_rel = y['joints_relations'].long().to(tgt.device)
        output = tgt
        if get_layer_activation > -1 and get_layer_activation < self.num_layers:
            activations=dict()
        for layer_ind, mod in enumerate(self.layers):
            edge_value_emb = None
            topology_value_emb = None
            if self.value_emb_flag:
                edge_value_emb = self.edge_value_emb
                topology_value_emb = self.topology_value_emb
            output = mod(
                    output, timesteps_embs, topology_rel, edge_rel, self.edge_key_emb, self.edge_query_emb, edge_value_emb, self.topology_key_emb, self.topology_query_emb, topology_value_emb, spatial_mask, temporal_mask,
                    tgt_key_padding_mask, memory_key_padding_mask, y,
                    source_memory=source_memory, tpose_bias=tpose_bias, src_cross_mask=src_cross_mask)
            if layer_ind == get_layer_activation:
                activations[layer_ind] = output.clone()
        if self.norm is not None:
            output = self.norm(output)
        if get_layer_activation > -1 and get_layer_activation < self.num_layers:
            return output, activations
        return output

class GraphMotionDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.d_model= d_model
        self.heads = nhead
        self.spatial_attn = GraphMultiHeadAttention(d_model = d_model, nheads = nhead, dropout=dropout)
        self.temporal_attn = MultiheadAttention(self.d_model, nhead, dropout=dropout)
        self.cross_motion_attn = CrossMotionAttention(d_model, nhead, dropout=dropout)
        self.norm_cross = nn.LayerNorm(d_model)
        self.dropout_cross = nn.Dropout(dropout)
        self.embed_timesteps = nn.Linear(d_model, d_model)

    # spatial attention block
    def _spatial_mha_block(self, x: Tensor, topology_rel: Optional[Tensor], edge_rel: Optional[Tensor], edge_key_emb, edge_query_emb, edge_value_emb,
        topology_key_emb, topology_query_emb, topology_value_emb, attn_mask: Optional[Tensor],  key_padding_mask: Optional[Tensor], y = None) -> Tensor:
        #x.shape (frames, bs, njoints, feature_len)
        frames, bs, njoints, feature_len = x.shape
        x = x.view(frames * bs, njoints, feature_len)
        topology_rel = topology_rel.unsqueeze(0).repeat(frames, 1, 1, 1).view(-1, njoints, njoints)
        edge_rel = edge_rel.unsqueeze(0).repeat(frames, 1, 1, 1).view(-1, njoints, njoints)
        
        attn_output = self.spatial_attn(x, x, x, topology_query_emb.weight, edge_query_emb.weight, topology_key_emb.weight, edge_key_emb.weight, None if topology_value_emb is None else topology_value_emb.weight, 
        None if edge_value_emb is None else edge_value_emb.weight, topology_rel, edge_rel, attn_mask)
        attn_output = attn_output.reshape(frames, bs, njoints, feature_len) # njoints, bs, frames, feature_len
        return self.dropout1(attn_output)
    
    
        # temporal attention block
    def _temporal_mha_block_sin_joint(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        frames, bs, njoints, feats= x.size() 
        # attn_mask_ = attn_mask[..., 1:, 1:]
        x = x.view(frames, bs * njoints, feats)
        output_attn, output_scores = self.temporal_attn(x, x, x,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask)
        output_attn = output_attn.view(frames, bs ,njoints, feats)
        return self.dropout2(output_attn)
    
    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

    def _cross_motion_attn_block(self, x: Tensor, source_memory: Tensor,
                                  tpose_bias: Tensor, src_cross_mask: Optional[Tensor]) -> Tensor:
        """
        Target motion이 source motion을 cross-attend한다.
        x:             [frames, bs, tgt_joints, d]
        source_memory: [frames, bs, src_joints, d]
        tpose_bias:    [bs, nheads, tgt_joints, src_joints]
        src_cross_mask:[bs, 1, 1, src_joints]
        """
        frames, bs, tgt_joints, d = x.shape
        src_joints = source_memory.shape[2]

        x_flat   = x.view(frames * bs, tgt_joints, d)
        src_flat = source_memory.view(frames * bs, src_joints, d)

        # T-pose bias: [bs, nheads, tgt_j, src_j] -> [frames*bs, nheads, tgt_j, src_j]
        tpose_bias_exp = tpose_bias.unsqueeze(0).expand(frames, -1, -1, -1, -1).reshape(
            frames * bs, self.heads, tgt_joints, src_joints)

        # src mask: [bs, 1, 1, src_j] -> [frames*bs, 1, 1, src_j]
        src_mask_exp = None
        if src_cross_mask is not None:
            src_mask_exp = src_cross_mask.unsqueeze(0).expand(frames, -1, -1, -1, -1).reshape(
                frames * bs, 1, 1, src_joints)

        out = self.cross_motion_attn(x_flat, src_flat, tpose_bias=tpose_bias_exp, src_mask=src_mask_exp)
        return self.dropout_cross(out.view(frames, bs, tgt_joints, d))

    def forward(self,
        tgt: Tensor,
        timesteps_emb: Tensor,
        topology_rel: Tensor,
        edge_rel: Tensor,
        edge_key_emb,
        edge_query_emb,
        edge_value_emb,
        topo_key_emb,
        topo_query_emb,
        topo_value_emb,
        spatial_mask: Optional[Tensor] = None,
        temporal_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None, #for future use
        y = None,
        source_memory: Optional[Tensor] = None,
        tpose_bias: Optional[Tensor] = None,
        src_cross_mask: Optional[Tensor] = None) -> Tensor:
        x = tgt #(frames, bs, njoints, feature_len)
        bs = x.shape[1]
        x = x + self.embed_timesteps(timesteps_emb).view(1, bs, 1, self.d_model)
        spatial_attn_output = self._spatial_mha_block(x, topology_rel, edge_rel, edge_key_emb, edge_query_emb, edge_value_emb,
        topo_key_emb, topo_query_emb, topo_value_emb, spatial_mask, tgt_key_padding_mask, y)
        x = self.norm1(x + spatial_attn_output)
        x = self.norm2(x + self._temporal_mha_block_sin_joint(x, temporal_mask, tgt_key_padding_mask))
        if source_memory is not None:
            x = self.norm_cross(x + self._cross_motion_attn_block(x, source_memory, tpose_bias, src_cross_mask))
        x = self.norm3(x + self._ff_block(x))
        return x
