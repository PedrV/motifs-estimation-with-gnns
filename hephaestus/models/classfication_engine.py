"""
Engine used to make predictions for motif fingerprints.
"""

import torch

from hephaestus.utils import general_utils as hutils


class ClassificationEngineV1(torch.nn.Module):
    def __init__(
        self,
        mpgnn,
        mpgnn_depth,
        mpgnn_hidden_dim,
        mpgnn_pool,
        mpgnn_dropout,
        mpgnn_jk,
        decoder,
        decoder_depth,
        decoder_hidden_dim,
        decoder_dropout,
        input_dim,
        output_dim,
        **kwargs
    ):
        super().__init__()

        if mpgnn_jk == "None":
            mpgnn_jk = None

        # kwargs["aggr"] = "max"
        # kwargs["v2"] = True  # Use GATv2
        # attention_heads = mpgnn_hidden_dim
        # if mpgnn_hidden_dim % 2 == 0:
        #     attention_heads = attention_heads // 2
        # elif mpgnn_hidden_dim % 3 == 0:
        #     attention_heads = attention_heads // 3
        # elif mpgnn_hidden_dim % 5 == 0:
        #     attention_heads = attention_heads // 5
        # kwargs["heads"] = attention_heads  # Number of heads GAT uses

        self.mpgnn = hutils.get_obj_from_str(mpgnn)(
            in_channels=input_dim,
            hidden_channels=mpgnn_hidden_dim,
            num_layers=mpgnn_depth,
            output_dim=mpgnn_hidden_dim,
            dropout=mpgnn_dropout,
            jk=mpgnn_jk,
            **kwargs
        )
        self.decoder = hutils.get_obj_from_str(decoder)(
            input_dim=mpgnn_hidden_dim,
            hidden_dim=decoder_hidden_dim,
            output_dim=output_dim,
            dropout=decoder_dropout,
            decoder_depth=decoder_depth,
            **kwargs
        )

        self.mpgnn_pool = hutils.get_obj_from_str(mpgnn_pool)

        self.my_versions = [
            ("engine", "V1"),
            ("decoder", self.decoder.my_version),
            ("gnn", mpgnn),
        ]

    def reset_parameters(self):
        self.mpgnn.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, edge_index, x, batch):
        # Info arrives already extracted from mini-batch
        embed = self.mpgnn(x, edge_index)
        graph_features = self.mpgnn_pool(embed, batch)
        out = self.decoder(graph_features)
        return out

    def return_gnn(self):
        return self.mpgnn

    def return_decoder(self):
        return self.decoder

    # tensor([[nan, nan, nan, nan, nan, nan, nan, nan]], dtype=torch.float64)
