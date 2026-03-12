import torch
torch.cuda.empty_cache()
import torch.nn as nn
from model.motion_transformer import GraphMotionDecoderLayer, GraphMotionDecoder, TposeCrossAttention


def create_sin_embedding(positions: torch.Tensor, dim: int, max_period: float = 10000,
                         dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create sinusoidal positional embedding, with shape `[B, T, C]`.

    Args:
        positions (torch.Tensor): LongTensor of positions.
        dim (int): Dimension of the embedding.
        max_period (float): Maximum period of the cosine/sine functions.
        dtype (torch.dtype or str): dtype to use to generate the embedding.
    Returns:
        torch.Tensor: Sinusoidal positional embedding.
    """
    # We aim for BTC format
    assert dim % 2 == 0
    half_dim = dim // 2
    positions = positions.to(dtype)
    adim = torch.arange(half_dim, device=positions.device, dtype=dtype).view(1, 1, -1)
    max_period_tensor = torch.full([], max_period, device=positions.device, dtype=dtype)  # avoid sync point
    phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)

class AnyTop(nn.Module):
    def __init__(self, max_joints, feature_len,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 activation="gelu", t5_out_dim = 512, root_input_feats=13,
                 **kargs):
        super().__init__()

        self.max_joints = max_joints
        self.feature_len = feature_len
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = self.feature_len
        self.root_input_feats = root_input_feats
        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.skip_t5=kargs.get('skip_t5', False)
        self.value_emb=kargs.get('value_emb', False)
        
        # target noisy motion => latent representation 
        self.input_process = InputProcess(self.input_feats, self.root_input_feats, self.latent_dim, t5_out_dim, skip_t5=self.skip_t5)
        # source motion => latent representation (for cross attention)
        self.source_input_process = InputProcess(self.input_feats, self.root_input_feats, self.latent_dim, t5_out_dim, skip_t5=self.skip_t5)
        self.tpose_cross_attn = TposeCrossAttention(self.latent_dim, self.num_heads)

        print("Graph transformer init")
        seqTransDecoderLayer = GraphMotionDecoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)
        self.seqTransDecoder = GraphMotionDecoder(seqTransDecoderLayer,
                                                        num_layers=self.num_layers, value_emb=self.value_emb)
            
        
        # latent (128) -> input feature (13)
        self.output_process = OutputProcess(self.feature_len, self.root_input_feats, self.max_joints, self.latent_dim)

    def forward(self, x, timesteps, get_layer_activation=-1, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        
        joints_mask = y['joints_mask'].to(x.device)
        temp_mask = y['mask'].to(x.device)
        tpos_first_frame = y['tpos_first_frame'].to(x.device).unsqueeze(0)

        bs, njoints, nfeats, nframes = x.shape
        timesteps_emb = create_sin_embedding(timesteps.view(1, -1, 1), self.latent_dim)[0]
        x = self.input_process(x, tpos_first_frame, y['joints_names_embs'], y['crop_start_ind']) # applies linear layer on each frame to convert it to latent dim
        spatial_mask = 1.0 - joints_mask[:, 0, 0, 1:, 1:]
        spatial_mask = spatial_mask.unsqueeze(1).unsqueeze(1).repeat(1, nframes + 1, self.num_heads, 1, 1).reshape(-1,self.num_heads, njoints, njoints)
        temporal_mask = 1.0 - temp_mask.repeat(1, njoints, self.num_heads, 1, 1).reshape(-1, nframes + 1, nframes + 1).float()
        spatial_mask[spatial_mask == 1.0] = -1e9
        temporal_mask[temporal_mask == 1.0] = -1e9

        # Source motion encoding + T-pose cross-attention
        source_memory = None
        tpose_bias = None
        src_cross_mask = None
        if 'source_motion' in y and y['source_motion'] is not None:
            src_motion = y['source_motion']                                     # [bs, src_joints, 13, nframes]
            src_tpos   = y['source_tpos_first_frame'].to(x.device).unsqueeze(0) # [1, bs, src_joints, 13]
            src_names  = y['source_joints_names_embs']                          # [bs, src_joints, t5_dim]
            src_crop   = y['source_crop_start_ind']                             # [bs]

            source_memory = self.source_input_process(src_motion, src_tpos, src_names, src_crop)
            # source_memory: [1+nframes, bs, src_joints, latent_dim]

            # T-pose embeddings: 각 InputProcess 출력의 첫 번째 프레임이 T-pose
            tgt_tpose_emb = x[0]             # [bs, tgt_joints, latent_dim]
            src_tpose_emb = source_memory[0] # [bs, src_joints, latent_dim]
            tpose_bias = self.tpose_cross_attn(tgt_tpose_emb, src_tpose_emb)
            # tpose_bias: [bs, nheads, tgt_joints, src_joints]

            # Source cross-attention mask: padded source joints를 -1e9로 마스킹
            src_n_joints  = y['source_n_joints'].to(x.device)  # [bs]
            src_max_joints = src_motion.shape[1]
            src_valid = torch.arange(src_max_joints, device=x.device).unsqueeze(0) < src_n_joints.unsqueeze(1)
            src_cross_mask = torch.zeros(bs, 1, 1, src_max_joints, device=x.device)
            src_cross_mask.masked_fill_(~src_valid.view(bs, 1, 1, src_max_joints), -1e9)

        # transformer block
        output = self.seqTransDecoder(tgt=x, timesteps_embs=timesteps_emb, memory=None, spatial_mask=spatial_mask, temporal_mask=temporal_mask, y=y, get_layer_activation=get_layer_activation,
                                      source_memory=source_memory, tpose_bias=tpose_bias, src_cross_mask=src_cross_mask)
        if get_layer_activation > -1 and get_layer_activation < self.num_layers:
            activations = output[1]
            output=output[0]
        output = self.output_process(output) # Applies linear layer on each frame to convert it back to feature len dim
        if get_layer_activation > -1 and get_layer_activation < self.num_layers:
            return output, activations
        return output


    def _apply(self, fn):
        super()._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)

# in the case of GMDM, the input process is as follows: 
# embed each joint of each frame of each motion in batch by the same MLP, separately ! 
class InputProcess(nn.Module):
    def __init__(self, input_feats, root_input_feats, latent_dim, t5_output_dim, skip_t5=False):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.root_input_feats = root_input_feats
        self.root_embedding = nn.Linear(self.root_input_feats, self.latent_dim)
        self.tpos_root_embedding = nn.Linear(self.root_input_feats, self.latent_dim)
        self.joint_embedding = nn.Linear(self.input_feats, self.latent_dim)
        self.tpos_joint_embedding = nn.Linear(self.input_feats, self.latent_dim)
        self.skip_t5=skip_t5
        if not self.skip_t5:
            self.joints_names_dropout = nn.Dropout(p=0.1)
            self.text_embedding = nn.Linear(t5_output_dim, self.latent_dim)
    
    def forward(self, x, tpos_first_frame, joints_embedded_names, crop_start_ind):
        # batch_size와 visualize sample의 크기가 다를경우
        # if tpos_first_frame.ndim == 4:
        #     tpos_first_frame = tpos_first_frame.squeeze(0)
            
        # x.shape = [batch_size, joints, 13, frames]
        x = x.permute(3, 0, 1, 2) # [frames, batch_size, n_joints, features_len]
        tpos_all_joints_except_root = self.tpos_joint_embedding(tpos_first_frame[:, :, 1:])
        tpos_root_data = self.tpos_root_embedding(tpos_first_frame[:, :, 0:1])
        all_joints_except_root = self.joint_embedding(x[:, :, 1:])
        root_data = self.root_embedding(x[:, :, 0:1])
        tpos_embedded = torch.cat([tpos_root_data, tpos_all_joints_except_root], dim=2)
        x_embedded = torch.cat([root_data, all_joints_except_root], dim=2) 
        x = torch.cat([tpos_embedded, x_embedded], dim=0)
        if not self.skip_t5:
            joints_embedded_names = self.text_embedding(self.joints_names_dropout(joints_embedded_names.to(x.device)))
            x = x + joints_embedded_names[None, ...]# [frames, batch_size, n_joints, d]
        positions = torch.arange(x.shape[0], device=x.device).view(1, -1, 1).repeat(x.shape[1], 1, 1)
        positions[:,1:,:] = positions[:,1:,:] + crop_start_ind.to(x.device).view(-1, 1, 1)
        pos_emb = create_sin_embedding(positions, self.latent_dim)[0]
        return x + pos_emb.unsqueeze(1).unsqueeze(1)

class OutputProcess(nn.Module):
    def __init__(self, feature_len, root_feature_len, max_joints, latent_dim):
        super().__init__()
        self.feature_len = feature_len
        self.max_joints = max_joints
        self.latent_dim = latent_dim
        self.root_feature_len = root_feature_len
        self.root_dembedding = nn.Linear(self.latent_dim, self.root_feature_len)
        self.joint_dembedding = nn.Linear(self.latent_dim, self.feature_len)

    def forward(self, output):
        # output shape [frames, batch_size, joints, latent_dim]
        root_data = self.root_dembedding(output[:, :, 0])
        all_joints = self.joint_dembedding(output[:, :, 1:])
        output = torch.cat([root_data.unsqueeze(2), all_joints], dim=-2)
        output = output.permute(1, 2, 3, 0)[..., 1:]  # [bs, njoints, nfeats, nframes]
        return output
