# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import torch
import numpy as np
import math
torch.autograd.set_detect_anomaly(True)

class PositionalEncoding(torch.nn.Module):
    """
    Ref: https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_length: int = 10000):
        """
        Args:
            d_model:      dimension of embeddings
            dropout:      randomly zeroes-out some of the input
            max_length:   max sequence length
        """
        # inherit from Module
        super().__init__()     

        # initialize dropout                  
        self.dropout = torch.nn.Dropout(p=dropout)      

        # create tensor of 0s
        pe = torch.zeros(max_length, d_model)    

        # create position column   
        k = torch.arange(0, max_length).unsqueeze(1)  

        # calc divisor for positional encoding 
        div_term = torch.exp(                                 
                torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        # calc sine on even indices
        pe[:, 0::2] = torch.sin(k * div_term)    

        # calc cosine on odd indices   
        pe[:, 1::2] = torch.cos(k * div_term)  

        # add dimension     
        pe = pe.unsqueeze(0)          

        # buffers are saved in state_dict but not trained by the optimizer                        
        self.register_buffer("pe", pe)                        

    def forward(self, x):
        """
        Args:
            x:        embeddings (batch_size, seq_length, d_model)

        Returns:
                    embeddings + positional encodings (batch_size, seq_length, d_model)
        """
        # add positional encoding to the embeddings
        x = x + self.pe[:, : x.size(1)].requires_grad_(False) 

        # perform dropout
        return self.dropout(x)

class TransformerSimple(torch.nn.Module):
    def __init__(self, model_config):
        super(TransformerSimple, self).__init__()

        # self.positional_encoding = PositionalEncoding(d_model=model_config['d_model'])
    
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=model_config['d_model'],
                                                        nhead=model_config['nhead'],
                                                        dim_feedforward=model_config['dim_feedforward'],
                                                        dropout=model_config['trans_dropout'],
                                                        batch_first=True)
                                                         
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer=encoder_layer,
                                                              num_layers=model_config['num_layers'])

        self.projection_head = torch.nn.Sequential(torch.nn.Linear(model_config['d_model'], model_config['h1']),
                                                  torch.nn.BatchNorm1d(model_config['h1']),
                                                  torch.nn.ReLU(),
                                                  torch.nn.Linear(model_config['h1'], model_config['embedding_size']))

    def forward(self, x):
        # x = self.transformer_encoder(x)
        # x = x.view(x.shape[0], -1)
        # x = self.projection_head(x)
        # return self.projection_head(self.transformer_encoder(self.positional_encoding(x).view(x.shape[0], -1)))
        return self.projection_head(self.transformer_encoder(x.view(x.shape[0], -1)))