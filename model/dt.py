import os, sys
sys.path.append(os.environ['DT_ROOT'])

import numpy as np
import torch
import torch.nn as nn

import constants

class DecisionTransformer(nn.Module):
    def __init__(self, emb_sz, timesteps, nhead, nlayers, device):
        super(DecisionTransformer, self).__init__()

        self.emb_sz = emb_sz
        transformer_layer = nn.TransformerEncoderLayer(d_model=emb_sz, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=nlayers)

        self.embed_t = nn.Embedding(timesteps, emb_sz)
        self.embed_s_a = nn.Embedding(constants.TOTAL, emb_sz)
        self.embed_R = nn.Linear(1, emb_sz)

        self.pred_a = nn.Linear(emb_sz, constants.TOTAL)

        self.layer_norm = nn.LayerNorm(emb_sz)

        self.device = device

    # Some code from https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py#L44
    def forward(self, inputs):
        R, s, a, t = inputs

        seq_length = R.shape[1]
        batch_size = R.shape[0]

        pos_embedding = self.embed_t(t)
        s_embedding = self.embed_s_a(s) + pos_embedding
        a_embedding = self.embed_s_a(a) + pos_embedding
        R_embedding = self.embed_R(torch.clamp(R.reshape(batch_size, seq_length, 1), -20, 0) / 20) + pos_embedding

        input_embeds = torch.stack((R_embedding, s_embedding, a_embedding), dim=1).permute(0, 2, 1, 3)\
            .reshape(batch_size, 3 * seq_length, self.emb_sz)

        input_embeds = self.layer_norm(input_embeds)

        mask_sz = 3 * seq_length
        attention_mask = torch.triu(torch.full((mask_sz, mask_sz), float('-inf'), device=self.device), diagonal=1)

        output_embeds = self.transformer_encoder(input_embeds, mask=attention_mask)
        output_processed = output_embeds.reshape(batch_size, seq_length, 3, self.emb_sz)\
            .permute(0, 2, 1, 3)

        return self.pred_a(output_processed[:,1,:,:])

