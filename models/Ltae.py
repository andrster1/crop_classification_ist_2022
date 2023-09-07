"""
Lightweight Temporal Attention Encoder module
Credits:
The module is heavily inspired by the works of Vaswani et al. on self-attention and their pytorch implementation of
the Transformer served as code base for the present script.
paper: https://arxiv.org/abs/1706.03762
code: github.com/jadore801120/attention-is-all-you-need-pytorch
"""

import torch
import torch.nn as nn
import numpy as np
import copy
from models.TCN import TemporalConvNet


"""
Arguments for the LTAE module:

parser.add_argument('--n_head', default=16, type=int, help='Number of attention heads')
parser.add_argument('--d_k', default=8, type=int, help='Dimension of the key and query vectors')
parser.add_argument('--mlp3', default='[256,128]', type=str, help='Number of neurons in the layers of MLP3')
parser.add_argument('--T', default=1000, type=int, help='Maximum period for the positional encoding')
parser.add_argument('--positions', default='bespoke', type=str,
                    help='Positions to use for the positional encoding (bespoke / order)')
parser.add_argument('--lms', default=24, type=int,
                    help='Maximum sequence length for positional encoding (only necessary if positions == order)')
parser.add_argument('--dropout', default=0.2, type=float, help='Dropout probability')
parser.add_argument('--d_model', default=256, type=int,
                    help="size of the embeddings (E), if input vectors are of a different size, a linear layer is used to project them to a d_model-dimensional space"
                    )
"""


class LTAE(nn.Module):
    def __init__(self, in_channels=128, n_head=16, d_k=8, n_neurons=[256,128], dropout=0.2, d_model=256,
                 T=1000, len_max_seq=24, positions=None, return_att=False):
        """
        Sequence-to-embedding encoder.
        Args:
            in_channels (int): Number of channels of the input embeddings
            n_head (int): Number of attention heads
            d_k (int): Dimension of the key and query vectors
            n_neurons (list): Defines the dimensions of the successive feature spaces of the MLP that processes
                the concatenated outputs of the attention heads
            dropout (float): dropout
            T (int): Period to use for the positional encoding
            len_max_seq (int, optional): Maximum sequence length, used to pre-compute the positional encoding table
            positions (list, optional): List of temporal positions to use instead of position in the sequence
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)
        """

        super(LTAE, self).__init__()
        self.in_channels = in_channels
        self.positions = positions
        self.n_neurons = copy.deepcopy(n_neurons)
        self.return_att = return_att

        if positions is None:
            positions = len_max_seq + 1

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Sequential(nn.Conv1d(in_channels, d_model, 1),
                                        nn.LayerNorm((d_model, len_max_seq)))
        else:
            self.d_model = in_channels
            self.inconv = None

        # sin_tab = get_sinusoid_encoding_table(positions, self.d_model // n_head, T=T)
        # self.position_enc = nn.Embedding.from_pretrained(torch.cat([sin_tab for _ in range(n_head)], dim=1),
        #                                                  freeze=True)


        #################################################################
        self.position_enc = PositionalEncoder(
            self.d_model // n_head, T=T, repeat=n_head
        )
        #################################################################


        self.inlayernorm = nn.LayerNorm(self.in_channels)

        self.outlayernorm = nn.LayerNorm(n_neurons[-1])

        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model, timesteps=len_max_seq)

        assert (self.n_neurons[0] == self.d_model)

        activation = nn.ReLU()

        layers = []
        for i in range(len(self.n_neurons) - 1):
            layers.extend([nn.Linear(self.n_neurons[i], self.n_neurons[i + 1]),
                           nn.BatchNorm1d(self.n_neurons[i + 1]),
                           activation])

        self.mlp = nn.Sequential(*layers)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, date=None, pad_mask=None):

        sz_b, seq_len, d = x.shape

        x = self.inlayernorm(x)

        if self.inconv is not None:
            x = self.inconv(x.permute(0, 2, 1)).permute(0, 2, 1)

        # if self.positions is None:
        #     src_pos = torch.arange(1, seq_len + 1, dtype=torch.long).expand(sz_b, seq_len).to(x.device)
        # else:
        #     src_pos = torch.arange(0, seq_len, dtype=torch.long).expand(sz_b, seq_len).to(x.device)
        # enc_output = x + self.position_enc(src_pos)

        ####################################################
        enc_output = x + self.position_enc(date)
        enc_output = enc_output.float()
        ####################################################T

        enc_output, attn = self.attention_heads(enc_output, enc_output, enc_output, pad_mask=pad_mask)

        enc_output = enc_output.permute(1, 0, 2).contiguous().view(sz_b, -1)  # Concatenate heads

        # enc_output = self.outlayernorm(self.dropout(self.mlp(enc_output)))
        enc_output = self.mlp(enc_output)

        if self.return_att:
            return enc_output, attn
        else:
            return enc_output


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_k, d_in, timesteps=24):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), timesteps=timesteps)

    def forward(self, q, k, v, pad_mask=None):
        #############################################
        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (self.n_head, 1)
            )
        ############################################
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = q.size()

        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(-1, d_k)  # (n*b) x d_k

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(n_head * sz_b, seq_len, -1)
        output, attn = self.attention(q, k, v, pad_mask)
        attn = attn.view(n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2)

        output = output.view(n_head, sz_b, 1, d_in // n_head)
        output = output.squeeze(dim=2)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1, timesteps=24):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.timesteps = timesteps

        """ For extra Pos embeddings"""
        # self.rel_pos_embs = AbsPosEmb1DAISummer(timesteps, 8)

    def forward(self, q, k, v, pad_mask=None):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature
        ####################################################
        if pad_mask is not None:
            pad_mask = pad_mask.bool()
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)
        ####################################################
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        output = torch.matmul(attn, v)

        return output, attn


def get_sinusoid_encoding_table(positions, d_hid, T=1000):
    ''' Sinusoid position encoding table
    positions: int or list of integer, if int range(positions)'''

    if isinstance(positions, int):
        positions = list(range(positions))

    def cal_angle(position, hid_idx):
        return position / np.power(T, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if torch.cuda.is_available():
        return torch.FloatTensor(sinusoid_table).cuda()
    else:
        return torch.FloatTensor(sinusoid_table)


def get_sinusoid_encoding_table_var(positions, d_hid, clip=4, offset=3, T=1000):
    ''' Sinusoid position encoding table
    positions: int or list of integer, if int range(positions)'''

    if isinstance(positions, int):
        positions = list(range(positions))

    x = np.array(positions)

    def cal_angle(position, hid_idx):
        return position / np.power(T, 2 * (hid_idx + offset // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])

    sinusoid_table = np.sin(sinusoid_table)  # dim 2i
    sinusoid_table[:, clip:] = torch.zeros(sinusoid_table[:, clip:].shape)

    if torch.cuda.is_available():
        return torch.FloatTensor(sinusoid_table).cuda()
    else:
        return torch.FloatTensor(sinusoid_table)

##################################################################################################

class TopPyramidLTAE(nn.Module):
    def __init__(self, in_channels=128, n_head=16, d_k=8, n_neurons=[256,128], dropout=0.2, d_model=256,
                 T=1000, len_max_seq=24, positions=None, return_att=False, q_steps=[12, 6, 1]):
        super().__init__()
        self.Ltaes = []
        for i in range(len(q_steps)):
            inter_positions = positions if i == 0 else None
            self.Ltaes.append(PyramidLTAE(in_channels, n_head, d_k, n_neurons, dropout, d_model,
                 T, len_max_seq//2**i, inter_positions, return_att, q_steps=q_steps[i]))
        self.Ltaes = nn.Sequential(*self.Ltaes)

    def forward(self, x):
        y = self.Ltaes(x)
        return y


class TopPyramidLTAE_V2(nn.Module):
    def __init__(self, in_channels=128, n_head=16, d_k=8, n_neurons=[256,128], dropout=0.2, d_model=256,
                 T=1000, len_max_seq=24, positions=None, return_att=False, q_steps=[12, 6, 1]):
        """
        Hierarchical LTAE with addition at residuals
        """
        super().__init__()

        self.n_neurons = n_neurons

        self.ltae1 = PyramidLTAE(in_channels, n_head, d_k, n_neurons, dropout, d_model,
                 T, len_max_seq, positions, return_att, q_steps=12)
        self.ltae2 = PyramidLTAE(in_channels, n_head, d_k, n_neurons, dropout, d_model,
                 T, len_max_seq//2, None, return_att, q_steps=6)
        self.ltae3 = PyramidLTAE(in_channels, n_head, d_k, n_neurons, dropout, d_model,
                 T, len_max_seq//4, None, return_att, q_steps=1)

        self.attn_weighting = VanillaAttention(n_head, d_k, in_channels)

        # self.final_mlp = nn.Sequential(
        #     nn.BatchNorm1d(in_channels),
        #     nn.Linear(in_channels, in_channels)
        # )

        self.bottleneck1 = TemporalConvNet(24, [12], kernel_size=4)
        self.bottleneck2 = TemporalConvNet(24, [12, 6], kernel_size=4)
        self.bottleneck3 = TemporalConvNet(24, [12, 6, 1], kernel_size=4)
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.norm3 = nn.LayerNorm(in_channels)

        layers = []
        for i in range(len(self.n_neurons) - 1):
            layers.extend(
                [
                    nn.Linear(self.n_neurons[i + 1], self.n_neurons[i + 1]),
                    nn.BatchNorm1d(self.n_neurons[i + 1]),
                    nn.ReLU()])

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.outlayernorm = nn.LayerNorm(n_neurons[-1])

        # they were not present for LTAE dimitris exp 1 and 2
        self.positions = positions
        self.inlayernorm = nn.LayerNorm(in_channels)
        sin_tab = get_sinusoid_encoding_table(positions, in_channels // n_head, T=T)
        self.vanilla_pos = nn.Embedding.from_pretrained(torch.cat([sin_tab for _ in range(n_head)], dim=1),
                                                         freeze=True)
        # if d_model is not None:
        #     self.d_model = d_model
        #     self.inconv = nn.Sequential(nn.Conv1d(in_channels, d_model, 1),
        #                                 nn.LayerNorm((d_model, len_max_seq)))
        # else:
        #     self.d_model = in_channels
        #     self.inconv = None

    def forward(self, x):
        sz_b, seq_len, d = x.shape

        # they were not present for LTAE dimitris exp 1 and 2
        # if self.inconv is not None:
        #     x = self.inconv(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.positions is None:
            src_pos = torch.arange(1, seq_len + 1, dtype=torch.long).expand(sz_b, seq_len).to(x.device)
        else:
            src_pos = torch.arange(0, seq_len, dtype=torch.long).expand(sz_b, seq_len).to(x.device)
        x = self.inlayernorm(x)
        x = x + self.vanilla_pos(src_pos)

        y_vanilla = self.attn_weighting(x, x, x)
        y1 = self.ltae1(y_vanilla) + self.norm1(self.bottleneck1(y_vanilla))
        y2 = self.ltae2(y1) + self.norm2(self.bottleneck2(y_vanilla))

        y3 = self.ltae3(y2) + self.norm3(self.bottleneck3(y_vanilla)).squeeze()
        y3 = self.outlayernorm(self.dropout(self.mlp(y3)))

        return y3


class TopPyramidLTAE_V3(nn.Module):

    def __init__(self, in_channels=128, n_head=16, d_k=8, n_neurons=[256,128], dropout=0.2, d_model=256,
                 T=1000, len_max_seq=24, positions=None, return_att=False, q_steps=[12, 6, 1]):
        """
        Hierarchical LTAE with concatenation at residuals
        """
        super().__init__()

        self.n_neurons = n_neurons
        self.ltae1 = PyramidLTAE(in_channels, n_head, d_k, n_neurons, dropout, d_model,
                 T, len_max_seq, positions, return_att, q_steps=12)
        self.ltae2 = PyramidLTAE(in_channels, n_head, d_k, n_neurons, dropout, d_model,
                 T, len_max_seq//2, None, return_att, q_steps=6)
        self.ltae3 = PyramidLTAE(in_channels, n_head, d_k, n_neurons, dropout, d_model,
                 T, len_max_seq//4, None, return_att, q_steps=1)

        self.attn_weighting = VanillaAttention(n_head, d_k, in_channels)

        # self.final_mlp = nn.Sequential(
        #     nn.BatchNorm1d(in_channels),
        #     nn.Linear(in_channels, in_channels)
        # )

        self.bottleneck1 = TemporalConvNet(36, [24, 12], kernel_size=4)
        self.bottleneck2 = TemporalConvNet(30, [24, 12, 6], kernel_size=4)
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

        layers = []
        for i in range(len(self.n_neurons) - 1):
            layers.extend(
                [
                    nn.Linear(self.n_neurons[i + 1], self.n_neurons[i + 1]),
                    nn.BatchNorm1d(self.n_neurons[i + 1]),
                    nn.ReLU()])

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.outlayernorm = nn.LayerNorm(n_neurons[-1])
        self.dropout = nn.Dropout(0.2)

        '''
        self.inlayernorm = nn.LayerNorm(self.in_channels)
        sin_tab = get_sinusoid_encoding_table(positions, d_model // n_head, T=T)
        self.vanilla_pos = nn.Embedding.from_pretrained(torch.cat([sin_tab for _ in range(n_head)], dim=1),
                                                         freeze=True)
        '''

    def forward(self, x):

        '''
        if self.positions is None:
            src_pos = torch.arange(1, seq_len + 1, dtype=torch.long).expand(sz_b, seq_len).to(x.device)
        else:
            src_pos = torch.arange(0, seq_len, dtype=torch.long).expand(sz_b, seq_len).to(x.device)
        x = self.inlayernorm(x)
        x = x + self.position_enc(src_pos)
        '''

        y_vanilla = self.attn_weighting(x, x, x)

        y1 = torch.cat((self.ltae1(y_vanilla), y_vanilla), 1)
        y1 = self.bottleneck1(self.norm1(self.dropout(y1)))

        y2 = torch.cat((self.ltae2(y1), y_vanilla), 1)
        y2 = self.bottleneck2(self.norm2(self.dropout(y2)))

        y3 = self.ltae3(y2)
        y3 = self.outlayernorm(self.dropout(self.mlp(y3)))
        return y3


class TopPyramidLTAE_CTC(nn.Module):
    def __init__(self, in_channels=128, n_head=16, d_k=8, n_neurons=[256,128], dropout=0.2, d_model=256,
                 T=1000, len_max_seq=24, positions=None, return_att=False, q_steps=[12, 6, 1]):
        super().__init__()

        self.ltae1 = PyramidLTAE(in_channels, n_head, d_k, n_neurons, dropout, d_model,
                 T, len_max_seq, positions, return_att, q_steps=12)
        self.ltae2 = PyramidLTAE(in_channels, n_head, d_k, n_neurons, dropout, d_model,
                 T, len_max_seq//2, None, return_att, q_steps=6)
        self.ltae3 = PyramidLTAE(in_channels, n_head, d_k, n_neurons, dropout, d_model,
                 T, len_max_seq//4, None, return_att, q_steps=1)

        self.ctc_mlp1 = nn.Linear(in_channels, 21)
        self.ctc_mlp2 = nn.Linear(in_channels, 21)

    def forward(self, x):
        y1 = self.ltae1(x)
        y2 = self.ltae2(y1)
        y3 = self.ltae3(y2)

        y1 = self.ctc_mlp1(y1)
        y2 = self.ctc_mlp2(y2)

        return y1, y2, y3


class TopPyramidLTAE_CTC_V2(nn.Module):
    def __init__(self, in_channels=128, n_head=16, d_k=8, n_neurons=[256,128], dropout=0.2, d_model=256,
                 T=1000, len_max_seq=24, positions=None, return_att=False, q_steps=[12, 6, 1]):
        super().__init__()

        self.positions = positions
        self.vanilla = VanillaAttention(n_head, d_k, in_channels)

        self.ltae1 = PyramidLTAE(in_channels, n_head, d_k, n_neurons, dropout, d_model,
                 T, len_max_seq, positions, return_att, q_steps=1)

        self.inlayernorm = nn.LayerNorm(in_channels)
        sin_tab = get_sinusoid_encoding_table(positions, in_channels // n_head, T=T)
        self.vanilla_pos = nn.Embedding.from_pretrained(torch.cat([sin_tab for _ in range(n_head)], dim=1),
                                                         freeze=True)

        self.ctc_mlp1 = nn.Linear(in_channels, 21)

    def forward(self, x):
        sz_b, seq_len, d = x.shape
        if self.positions is None:
            src_pos = torch.arange(1, seq_len + 1, dtype=torch.long).expand(sz_b, seq_len).to(x.device)
        else:
            src_pos = torch.arange(0, seq_len, dtype=torch.long).expand(sz_b, seq_len).to(x.device)
        x = self.inlayernorm(x)
        x = x + self.vanilla_pos(src_pos)

        y1 = self.vanilla(x, x, x)
        y2 = self.ltae1(y1)

        y_ctc = self.ctc_mlp1(y1)

        return y_ctc, y2


class TopPyramidLTAE_CTC_V3_Residuals(nn.Module):
    def __init__(self, in_channels=128, n_head=16, d_k=8, n_neurons=[256,128], dropout=0.2, d_model=256,
                 T=1000, len_max_seq=24, positions=None, return_att=False, q_steps=[12, 6, 1]):
        super().__init__()

        self.ltae1 = PyramidLTAE(in_channels, n_head, d_k, n_neurons, dropout, d_model,
                 T, len_max_seq, positions, return_att, q_steps=12)
        self.ltae2 = PyramidLTAE(in_channels, n_head, d_k, n_neurons, dropout, d_model,
                 T, len_max_seq//2, None, return_att, q_steps=6)
        self.ltae3 = PyramidLTAE(in_channels, n_head, d_k, n_neurons, dropout, d_model,
                 T, len_max_seq//4, None, return_att, q_steps=1)

        self.ctc_mlp1 = nn.Linear(in_channels, 21)
        self.ctc_mlp2 = nn.Linear(in_channels, 21)

        self.bottleneck1 = TemporalConvNet(12, [6], kernel_size=3)
        self.bottleneck2 = TemporalConvNet(6, [1], kernel_size=3)
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

    def forward(self, x):
        y1 = self.ltae1(x)

        y2 = self.ltae2(y1)
        y2_res = y2 + self.norm1(self.bottleneck1(y1))

        y3 = self.ltae3(y2_res)
        y3_res = y3 + self.norm2(self.bottleneck2(y2_res)).squeeze()

        y1_ctc = self.ctc_mlp1(y1)
        y2_ctc = self.ctc_mlp2(y2)

        return y1_ctc, y2_ctc, y3_res


class TopPyramidLTAE_CTC_V4_TCN_Query(nn.Module):
    def __init__(self, in_channels=128, n_head=16, d_k=8, n_neurons=[256,128], dropout=0.2, d_model=256,
                 T=1000, len_max_seq=24, positions=None, return_att=False, q_steps=[12, 6, 1]):
        super().__init__()

        self.positions = positions
        self.vanilla = VanillaAttention(n_head, d_k, in_channels)

        self.ltae1 = PyramidLTAE(in_channels, n_head, d_k, n_neurons, dropout, d_model,
                 T, len_max_seq, positions, return_att, q_steps=1)

        self.inlayernorm = nn.LayerNorm(in_channels)
        sin_tab = get_sinusoid_encoding_table(positions, in_channels // n_head, T=T)
        self.vanilla_pos = nn.Embedding.from_pretrained(torch.cat([sin_tab for _ in range(n_head)], dim=1),
                                                         freeze=True)

        self.ctc_mlp1 = nn.Linear(in_channels, 21)

        self.query_production = TemporalConvNet(24, [12, 6, 1], kernel_size=4)

    def forward(self, x):
        sz_b, seq_len, d = x.shape
        if self.positions is None:
            src_pos = torch.arange(1, seq_len + 1, dtype=torch.long).expand(sz_b, seq_len).to(x.device)
        else:
            src_pos = torch.arange(0, seq_len, dtype=torch.long).expand(sz_b, seq_len).to(x.device)
        x = self.inlayernorm(x)
        x = x + self.vanilla_pos(src_pos)

        y1 = self.vanilla(x, x, x)

        query = self.query_production(y1)
        y2 = self.ltae1(y1, query)

        y_ctc = self.ctc_mlp1(y1)

        return y_ctc, y2


class VanillaAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.fc1_q = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_q.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        # self.fc1_v = nn.Linear(d_in, n_head * d_k)
        # nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc2 = nn.Sequential(
            nn.BatchNorm1d(d_in),
            nn.Linear(d_in, n_head * d_k)
        )

        self.attention = PyramidScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, q, k, v):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = q.size()

        q = self.fc1_q(q).view(sz_b, seq_len, n_head, d_k)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        k = self.fc1_k(k).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        # v = self.fc1_v(v).view(sz_b, seq_len, n_head, d_k)
        # v = v.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(n_head * sz_b, seq_len, -1)

        output, attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, -1, d_in // n_head)
        # output = output.squeeze(dim=2)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, -1, n_head * d_in // n_head).squeeze()

        return output


class PyramidLTAE(nn.Module):
    def __init__(self, in_channels=128, n_head=16, d_k=8, n_neurons=[256, 128], dropout=0.2, d_model=256,
                 T=1000, len_max_seq=24, positions=None, return_att=False, q_steps=1):
        """
        Sequence-to-embedding encoder.
        Args:
            in_channels (int): Number of channels of the input embeddings
            n_head (int): Number of attention heads
            d_k (int): Dimension of the key and query vectors
            n_neurons (list): Defines the dimensions of the successive feature spaces of the MLP that processes
                the concatenated outputs of the attention heads
            dropout (float): dropout
            T (int): Period to use for the positional encoding
            len_max_seq (int, optional): Maximum sequence length, used to pre-compute the positional encoding table
            positions (list, optional): List of temporal positions to use instead of position in the sequence
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)
        """

        super().__init__()
        self.in_channels = in_channels
        self.positions = positions
        self.n_neurons = copy.deepcopy(n_neurons)
        self.return_att = return_att
        self.q_steps = q_steps

        if positions is None:
            positions = len_max_seq + 1

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Sequential(nn.Conv1d(in_channels, d_model, 1),
                                        nn.LayerNorm((d_model, len_max_seq)))
        else:
            self.d_model = in_channels
            self.inconv = None

        sin_tab = get_sinusoid_encoding_table(positions, self.d_model // n_head, T=T)
        self.position_enc = nn.Embedding.from_pretrained(torch.cat([sin_tab for _ in range(n_head)], dim=1),
                                                         freeze=True)

        self.inlayernorm = nn.LayerNorm(self.in_channels)

        self.attention_heads = PyramidMultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model, q_steps=q_steps, timesteps=len_max_seq)

        assert (self.n_neurons[0] == self.d_model)

        activation = nn.ReLU()

        layers = []
        for i in range(len(self.n_neurons) - 1):
            layers.extend([nn.Linear(self.n_neurons[i], self.n_neurons[i + 1]),
                           nn.LayerNorm((self.n_neurons[i + 1])),
                           activation])

        self.mlp = nn.Sequential(*layers)

        self.outlayernorm = nn.LayerNorm(n_neurons[-1])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, query=None):

        sz_b, seq_len, d = x.shape

        x = self.inlayernorm(x)

        if self.inconv is not None:
            x = self.inconv(x.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positions is None:
            src_pos = torch.arange(1, seq_len + 1, dtype=torch.long).expand(sz_b, seq_len).to(x.device)
        else:
            src_pos = torch.arange(0, seq_len, dtype=torch.long).expand(sz_b, seq_len).to(x.device)
        enc_output = x + self.position_enc(src_pos)

        if query is None:
            enc_output, attn = self.attention_heads(enc_output, enc_output, enc_output)
        else:
            enc_output, attn = self.attention_heads(query, enc_output, enc_output)

        enc_output = enc_output.permute(1, 2, 0, 3).contiguous().view(sz_b, self.q_steps, -1).squeeze()  # Concatenate heads

        enc_output = self.outlayernorm(self.dropout(self.mlp(enc_output)))
        ### Pyramid LTAE V3 has droput+norm at the concatenated features ###
        # enc_output = self.mlp(enc_output)

        if self.return_att:
            return enc_output, attn
        else:
            return enc_output


class PyramidMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_k, d_in, q_steps, timesteps=24):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in
        self.q_steps = q_steps
        self.Q = nn.Parameter(torch.zeros((q_steps, n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = PyramidScaledDotProductAttention(temperature=np.power(d_k, 0.5), timesteps=timesteps)

    def forward(self, q, k, v):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = k.size()

        q = torch.stack([self.Q for _ in range(sz_b)], dim=-1)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, self.q_steps, d_k)

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(n_head * sz_b, seq_len, -1)
        output, attn = self.attention(q, k, v)
        attn = attn.view(n_head, sz_b, self.q_steps, seq_len)
        attn = attn.squeeze(dim=2)

        output = output.view(n_head, sz_b, self.q_steps, d_in // n_head)
        output = output#.squeeze(dim=2)

        return output, attn


class PyramidScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1, timesteps=24):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.timesteps = timesteps

    def forward(self, q, k, v):

        attn = torch.matmul(q, k.transpose(1, 2))
        attn = attn / self.temperature
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        output = torch.matmul(attn, v)

        return output, attn

##################################################################################################

class LTAE_kosmas_diffs(nn.Module):
    def __init__(self, in_channels=128, n_head=16, d_k=8, n_neurons=[256,128], dropout=0.2, d_model=256,
                 T=1000, len_max_seq=24, positions=None, return_att=False):
        """
        Sequence-to-embedding encoder.
        Args:
            in_channels (int): Number of channels of the input embeddings
            n_head (int): Number of attention heads
            d_k (int): Dimension of the key and query vectors
            n_neurons (list): Defines the dimensions of the successive feature spaces of the MLP that processes
                the concatenated outputs of the attention heads
            dropout (float): dropout
            T (int): Period to use for the positional encoding
            len_max_seq (int, optional): Maximum sequence length, used to pre-compute the positional encoding table
            positions (list, optional): List of temporal positions to use instead of position in the sequence
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)
        """

        super().__init__()
        self.in_channels = in_channels
        self.positions = positions
        self.n_neurons = copy.deepcopy(n_neurons)
        self.return_att = return_att


        if positions is None:
            positions = len_max_seq + 1

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Sequential(nn.Conv1d(in_channels, d_model, 1),
                                        nn.LayerNorm((d_model, len_max_seq)))
        else:
            self.d_model = in_channels
            self.inconv = None

        # sin_tab = get_sinusoid_encoding_table(positions, self.d_model // n_head, T=T)
        # self.position_enc = nn.Embedding.from_pretrained(torch.cat([sin_tab for _ in range(n_head)], dim=1),
        #                                                  freeze=True)


        self.position_enc = PositionalEncoder(
            self.d_model // n_head, T=T, repeat=n_head
        )


        self.inlayernorm = nn.LayerNorm(self.in_channels)

        self.outlayernorm = nn.LayerNorm(n_neurons[-1])

        self.attention_heads = MultiHeadAttention_kosmas_diffs(
            n_head=n_head, d_k=d_k, d_in=self.d_model, timesteps=len_max_seq)

        # assert (self.n_neurons[0] == self.d_model)

        activation = nn.ReLU()

        layers = []
        for i in range(len(self.n_neurons) - 1):
            layers.extend([nn.Linear(self.n_neurons[i], self.n_neurons[i + 1]),
                           nn.BatchNorm1d(self.n_neurons[i + 1]),
                           activation])

        self.mlp = nn.Sequential(*layers)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, dates):

        sz_b, seq_len, d = x.shape

        x = self.inlayernorm(x)

        if self.inconv is not None:
            x = self.inconv(x.permute(0, 2, 1)).permute(0, 2, 1)

        # if self.positions is None:
        #     src_pos = torch.arange(1, seq_len + 1, dtype=torch.long).expand(sz_b, seq_len).to(x.device)
        # else:
        #     src_pos = torch.arange(0, seq_len, dtype=torch.long).expand(sz_b, seq_len).to(x.device)
        # enc_output = x + self.position_enc(src_pos)

        if self.positional_encoder is not None:
            bp = (
                dates.unsqueeze(-1)
                .repeat((1, 1, d))
            )  # BxTxd
            try:
                bp = bp.permute(0, 2, 1).contiguous().view(sz_b * d, seq_len)
            except:
                print()

        enc_output = x + self.positional_encoder(bp)

        k = self.attention_heads(enc_output, enc_output, enc_output)

        return k


class MultiHeadAttention_kosmas_diffs(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_k, d_in, timesteps=24):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

    def forward(self, q, k, v):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = q.size()

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        return k

################################################################

def mask(length : int, window_size : int):
  x = torch.zeros(length, length).to(dtype=torch.long)
  for i in range(x.size(0)):
    x[i, i:i+window_size] = 1
  return x



#######################################################
class PositionalEncoder(nn.Module):
    def __init__(self, d, T=1000, repeat=None, offset=0):
        super(PositionalEncoder, self).__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        self.denom = torch.pow(
            T, 2 * (torch.arange(offset, offset + d).float() // 2) / d
        )
        self.updated_location = False

    def forward(self, batch_positions):
        if not self.updated_location:
            self.denom = self.denom.to(batch_positions.device)
            self.updated_location = True
        sinusoid_table = (
            batch_positions[:, :, None] / self.denom[None, None, :]
        )  # B x T x C
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        if self.repeat is not None:
            sinusoid_table = torch.cat(
                [sinusoid_table for _ in range(self.repeat)], dim=-1
            )

        return sinusoid_table