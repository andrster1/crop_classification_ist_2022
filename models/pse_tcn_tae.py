import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pse import PixelSetEncoder
from models.tae import TemporalAttentionEncoder
from models.decoder import get_decoder


class PseTCNTae(nn.Module):
    def __init__(self, input_dim=10, mlp1=[10, 32, 64], pooling='mean_std', mlp2=[132, 128], with_extra=True,
                 extra_size=4, n_head=4, d_k=32, d_model=None, mlp3=[512, 128, 128], dropout=0.2, T=1000,
                 len_max_seq=24, mlp4=[128, 64, 32, 20], positions=None):
        super(PseTCNTae, self).__init__()
        self.n_head = n_head
        self.len_max_seq = len_max_seq
        self.d_model = d_model
        self.d_k = d_k
        self.spatial_encoder = PixelSetEncoder(input_dim, mlp1=mlp1, pooling=pooling, mlp2=mlp2, with_extra=with_extra,
                                               extra_size=extra_size)
        self.diff_spatial_encoder = PixelSetEncoder(input_dim, mlp1=mlp1, pooling=pooling, mlp2=mlp2, with_extra=with_extra,
                                               extra_size=extra_size)
        self.temporal_encoder = TemporalAttentionEncoder(in_channels=mlp2[-1], n_head=n_head, d_k=d_k, d_model=d_model,
                                                         n_neurons=mlp3, dropout=dropout,
                                                         T=T, len_max_seq=24, positions=positions)
        self.diff_temporal_encoder = TemporalAttentionEncoder(in_channels=mlp2[-1], n_head=n_head, d_k=d_k, d_model=d_model,
                                                         n_neurons=mlp3, dropout=dropout,
                                                         T=T, len_max_seq=23, tcn_in=23, positions=None)
        self.decoder = get_decoder(mlp4)
        self.outlayernorm = nn.BatchNorm1d(mlp4[0])
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, diff):
        """
         Args:
            input(tuple): ((Pixel-Set, Pixel-Mask), Extra-features)
            Pixel-Set: Batch_size x Sequence length x Channel x Number of pixels
            Pixel-Mask: Batch_size x Sequence length x Number of pixels
            Extra-features: Batch_size x Sequence length x Number of features
            diff(tuple): ((Pixel-Set, Pixel-Mask), Extra-features)
            date: dates for each timestep that were captured
        """
        sz_b = input[0][0].size(0)
        out = self.spatial_encoder([[input[0][0], input[0][1]], input[1]])
        out = self.temporal_encoder(out)

        out_diff = self.diff_spatial_encoder([[diff[0][0], diff[0][1]], diff[1]])
        diff_pos = torch.arange(1, 24).to(out_diff.device)[None]
        diff_pos = diff_pos.repeat(sz_b, 1)
        out_diff = self.diff_temporal_encoder(out_diff, diff_pos)

        out_concated = torch.cat([out, out_diff], -1)
        out_concated = F.relu(self.outlayernorm(self.dropout(out_concated)))
        out = self.decoder(out_concated)
        return out

    def param_ratio(self):
        total = get_ntrainparams(self)
        s = get_ntrainparams(self.spatial_encoder)
        t = get_ntrainparams(self.temporal_encoder)
        c = get_ntrainparams(self.decoder)

        print('TOTAL TRAINABLE PARAMETERS : {}'.format(total))
        print('RATIOS: Spatial {:5.1f}% , Temporal {:5.1f}% , Classifier {:5.1f}%'.format(s / total * 100,
                                                                                          t / total * 100,
                                                                                          c / total * 100))


def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
