import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder

class CustomTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super(CustomTransformerEncoderLayer, self).__init__(*args, **kwargs)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                            key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights

class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(CustomTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        attention_weights = []

        for layer in self.layers:
            output, attn_weights = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attention_weights.append(attn_weights)

        return output, attention_weights


class SubFormer_Prot(nn.Module):

    def __init__(self,hidden_channels, encoder_layers, nhead, dim_feedforward, dropout=0.1, attn_map=True):
        super(SubFormer_Prot, self).__init__()

        ## transformer encoder
        self.attn_map = attn_map
        if attn_map:
            
            encoder_layer = CustomTransformerEncoderLayer(batch_first=True,
                                                    d_model=hidden_channels,
                                                    nhead=nhead, 
                                                    dim_feedforward=dim_feedforward, 
                                                    dropout=dropout)

            self.transformer_encoder = CustomTransformerEncoder(encoder_layer, num_layers=encoder_layers)
        else:
            encoder_layer = TransformerEncoderLayer(batch_first=True,
                                                    d_model=hidden_channels, 
                                                    nhead=nhead, 
                                                    dim_feedforward=dim_feedforward, 
                                                    dropout=dropout)

            self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=encoder_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_channels))

    
    def get_weights(self,data):

        graph_emb = data
        src = torch.cat((self.cls_token.expand(graph_emb.size(0), -1, -1), graph_emb), dim=1)
        src, attention_weights = self.transformer_encoder(src, src_key_padding_mask=None)
        return attention_weights
    

    def forward(self, data):

        src = data
        src = torch.cat((self.cls_token.expand(src.size(0), -1, -1), src), dim=1)
        if self.attn_map:
            src, attention_weights = self.transformer_encoder(src, src_key_padding_mask=None)
        else:
            src = self.transformer_encoder(src, src_key_padding_mask=None)

        out_token = src[:,0,:]

        return out_token

