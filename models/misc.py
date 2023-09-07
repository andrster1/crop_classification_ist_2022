"""
Temporal Attention Encoder module

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


class TemporalAttentionEncoder(nn.Module):
    def __init__(self, in_channels=128, n_head=4, d_k=32, d_model=None, n_neurons=[512, 128, 128], dropout=0.2,
                 T=1000, len_max_seq=24, positions=None, tcn_in=24):
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

        """

        super(TemporalAttentionEncoder, self).__init__()
        self.in_channels = in_channels
        self.positions = positions
        self.n_neurons = copy.deepcopy(n_neurons)

        self.name = 'TAE_dk{}_{}Heads_{}_T{}_do{}'.format(d_k, n_head, '|'.join(list(map(str, self.n_neurons))), T,
                                                          dropout)

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Sequential(nn.Conv1d(in_channels, d_model, 1),
                                        nn.LayerNorm((d_model, len_max_seq))
                                        )
            self.name += '_dmodel{}'.format(d_model)
        else:
            self.d_model = in_channels
            self.inconv = None

        if positions is None:
            positions = len_max_seq + 1
        else:
            self.name += '_bespokePos'

        # self.position_enc = nn.Embedding.from_pretrained(
        #     get_sinusoid_encoding_table(positions, self.in_channels, T=T),
        #     freeze=True)

        sin_tab = get_sinusoid_encoding_table(positions, self.d_model // n_head, T=T)
        self.position_enc = nn.Embedding.from_pretrained(torch.cat([sin_tab for _ in range(n_head)], dim=1),
                                                         freeze=True)



        self.inlayernorm = nn.LayerNorm(self.in_channels)

        self.outlayernorm = nn.LayerNorm(n_neurons[-1])

        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model, tcn_in=tcn_in)

        # assert (self.n_neurons[0] == self.d_model)
        # assert (self.n_neurons[-1] == self.d_model)
        layers = []
        for i in range(len(self.n_neurons) - 1):
            layers.extend(
                [
                    nn.Linear(self.n_neurons[i], self.n_neurons[i + 1]),
                           nn.BatchNorm1d(self.n_neurons[i + 1]),
                           nn.ReLU()])

        self.mlp = nn.Sequential(*layers)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, date=None, pad_mask=None):

        sz_b, seq_len, d = x.shape

        x = self.inlayernorm(x)

        if self.inconv is not None:
            x = self.inconv(x.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positions is None:
            src_pos = torch.arange(1, seq_len + 1, dtype=torch.long).expand(sz_b, seq_len).to(x.device)
        else:
            src_pos = torch.arange(0, seq_len, dtype=torch.long).expand(sz_b, seq_len).to(x.device)
        enc_output = x + self.position_enc(src_pos)

        # fff = self.position_enc(src_pos)

        enc_output, attn = self.attention_heads(enc_output, enc_output, enc_output, pad_mask=pad_mask)

        """ commented for q concat in self attention experiment"""
        # enc_output = enc_output.permute(1, 0, 2).contiguous().view(sz_b, -1)  # Concatenate heads

        # enc_output = self.outlayernorm(self.dropout(self.mlp(enc_output)))
        # enc_output = self.outlayernorm(self.dropout(self.mlp(enc_output[0]))) + enc_output[1]
        """ I commented above line for the differences experiments. 
        I do dropout and normalization at the concated outputs"""
        # enc_output = self.mlp(enc_output)

        return torch.cat((enc_output[0], enc_output[1]), -1)


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_k, d_in, tcn_in=24):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        # self.fc1_q = nn.Linear(d_in, n_head * d_k)

        """ ID convs to reduce Q instead of mean """
        # self.fc1_q = nn.Sequential(
        #             nn.Conv1d(128, 256, 4, 2),
        #             nn.BatchNorm1d(256),
        #             nn.ReLU(),
        #             nn.AvgPool1d(2, 2),
        #             nn.Conv1d(256, 128, 4, 2),
        #             nn.BatchNorm1d(128),
        #             nn.ReLU(),
        #         )

        self.fc1_q = TemporalConvNet(tcn_in, [12, 6, 3, 1], kernel_size=4)

        # nn.init.normal_(self.fc1_q.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc2 = nn.Sequential(
            nn.BatchNorm1d(d_in),
            nn.Linear(d_in, n_head * d_k)
        )

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

        ####
        # self.same = nn.Sequential(
        #     nn.Linear(n_head * d_k, d_in),
        #     nn.BatchNorm1d(d_in),
        #     nn.ReLU()
        # )

    def forward(self, q, k, v, pad_mask=None):
        #############################################
        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (self.n_head, 1)
            )
        ############################################

        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = q.size()

        q = self.fc1_q(q).squeeze()
        # q = q.mean(dim=1).squeeze()  # MEAN query
        q_orig = self.fc2(q)
        # q_orig = q
        q = q_orig.view(sz_b, n_head, d_k).permute(1, 0, 2).contiguous().view(n_head * sz_b, d_k)

        k = self.fc1_k(k).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        # v = v.repeat(n_head, 1, 1)  # (n*b) x lv x d_in
        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(n_head * sz_b, seq_len, -1)

        output, attn = self.attention(q, k, v, pad_mask)

        # output = output.view(n_head, sz_b, 1, d_in)
        output = output.view(n_head, sz_b, 1, d_in // n_head)
        # output = output.squeeze(dim=2)
        """ concat q exp """
        output = output.squeeze(dim=2).permute(1, 0, 2).contiguous().view(sz_b, -1)

        # return torch.cat((output, q_orig), -1), attn

        #######
        # q_orig = self.same(q_orig)
        return(output, q_orig), attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

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





