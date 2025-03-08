"""
Decoder module used in classification_engine.py
"""

import torch


class SimpleDecoder(torch.nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, dropout=0.5, decoder_depth=3, **kwargs
    ):
        super().__init__()

        self.my_version = "simple-decoder_V1"
        if "my_decoder_version" in kwargs:
            self.my_version = kwargs["my_decoder_version"]

        self.dropout = dropout
        self.decoder_depth = decoder_depth

        self.linears = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()

        if decoder_depth == 1:
            self.linears.append(torch.nn.Linear(input_dim, output_dim))
            self.activations.append(torch.nn.Tanh())
        else:
            self.linears.append(torch.nn.Linear(input_dim, hidden_dim))
            self.activations.append(torch.nn.Tanh())
            for _ in range(self.decoder_depth - 2):
                self.linears.append(torch.nn.Linear(hidden_dim, hidden_dim))
                self.activations.append(torch.nn.Tanh())
                self.dropouts.append(torch.nn.Dropout(self.dropout))
            self.linears.append(torch.nn.Linear(hidden_dim, output_dim))
            self.activations.append(torch.nn.Tanh())

    def reset_parameters(self):
        for i in range(len(self.linears)):
            self.linears[i].reset_parameters()

    def forward(self, graph_features):
        out = graph_features
        # https://discuss.pytorch.org/t/batch-processing-in-linear-layers/77527
        for i in range(len(self.linears)):
            out = self.activations[i](self.linears[i](out))
            if i > 0 and i < len(self.linears) - 1:
                out = self.dropouts[i - 1](out)

        if self.my_version == "simple-decoder_V2":
            n0 = torch.linalg.norm(out[:, 0:2], dim=1)
            n1 = torch.linalg.norm(out[:, 2:], dim=1)
            out2 = out.clone()
            out2[:, :2] /= n0.view(n0.size()[0], 1)
            out2[:, 2:] /= n1.view(n0.size()[0], 1)
            return out2

        return out
