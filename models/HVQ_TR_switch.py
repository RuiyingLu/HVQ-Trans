import time

import lmdb
import numpy as np
import torch
import random
from torch import nn
from torch.nn import functional as F

import distributed as dist_fn
from torchvision import models
from models.backbones import efficientnet_b4
from models.transformer import TransformerDecoder_hierachy, Org_TransformerDecoderLayer, build_position_embedding
from models.transformer import TransformerEncoder, TransformerEncoderLayer
from einops import rearrange


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class HVQ_TR_switch(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
    ):
        super().__init__()

        # Encoder
        self.enc = efficientnet_b4(pretrained=True, outblocks=[1,5,9,21],outstrides=[2,4,8,16])
        for k, p in self.enc.named_parameters():
            p.requires_grad = False

        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)

        # 15 codebook
        self.quantize_list_1 = nn.ModuleList([])
        self.quantize_list_2 = nn.ModuleList([])
        self.quantize_list_3 = nn.ModuleList([])
        self.quantize_list_4 = nn.ModuleList([])
        self.quantize_list_5 = nn.ModuleList([])
        for i in range(15):
            self.quantize_list_1.append(Quantize(embed_dim*2, 512))
            self.quantize_list_2.append(Quantize(embed_dim*2, 512))
            self.quantize_list_3.append(Quantize(embed_dim*2, 512))
            self.quantize_list_4.append(Quantize(embed_dim*2, 512))
            self.quantize_list_5.append(Quantize(embed_dim, 512))

        # Encoder
        encoder_layer = TransformerEncoderLayer(
            embed_dim, 8, dim_feedforward=1024, dropout=0.1, activation='relu', normalize_before=False
        )
        encoder_norm = None
        self.encoder = TransformerEncoder(
            encoder_layer, 4, encoder_norm, return_intermediate=True, return_src=True
        )

        # Decoder
        self.feature_size = (14, 14)
        self.pos_embed = build_position_embedding(
            'learned', self.feature_size, embed_dim
        )

        decoder_layer = Org_TransformerDecoderLayer(
            embed_dim * 2,
            self.feature_size,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation='relu',
            normalize_before=False,
        )

        decoder_norm = nn.LayerNorm(embed_dim * 2)
        self.decoder = TransformerDecoder_hierachy(
            decoder_layer,
            4,
            decoder_norm,
            return_intermediate=False,
        )
        self.input_proj = nn.Linear(channel, embed_dim)
        # self.output_proj = nn.Linear(embed_dim * 2, channel)
        self.output_proj_list = nn.ModuleList([])
        for i in range(15):
            self.output_proj_list.append(nn.Linear(embed_dim * 2, channel))

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=16)
        self.feature_loss = nn.MSELoss()
        self.rec_loss = nn.MSELoss()
        self.latent_loss_weight = 0.25
        self.channel = channel

        self.scale_factors = [0.125, 0.25, 0.5, 1.0]
        self.upsample_list = [
            nn.UpsamplingBilinear2d(scale_factor=scale_factors)
            for scale_factors in self.scale_factors
        ]

    def forward(self, inputs):
        input = inputs['image']
        label = inputs['clslabel']

        org_feature = self.extract_feature(inputs)

        feature_tokens = self.quantize_conv_t(org_feature)
        feature_tokens = rearrange(
            feature_tokens, "b c h w -> (h w) b c"
        )  # (H x W) x B x C
        L, batch_size, C = feature_tokens.size()

        # pos embed
        pos_embed = self.pos_embed(feature_tokens)
        pos_embed = torch.cat(
            [pos_embed.unsqueeze(1)] * batch_size, dim=1
        )   # (H x W) x B x C

        # Enocde
        output_encoder = self.encoder(feature_tokens,mask=None,pos=pos_embed)   # (5, 196, Bs, 256)
        output_encoder = rearrange(output_encoder, 'n l b c -> n b l c')

        # VQ
        quant_list, diff, _ = self.encode(output_encoder, label)    # (4, Bs, 196, 512)
        quant_list = rearrange(quant_list, 'n b l c -> n l b c')    # (4, 196, Bs, 512)
        decode_pos_embed = torch.cat([pos_embed, pos_embed],dim=2)  # (196, Bs, 512)

        dec = self.decoder(
            quant_list,
            tgt_mask=None,
            memory_mask=None,
            pos=decode_pos_embed
        )

        # switch
        feature_rec_tokens = torch.zeros(size=(L, batch_size, self.channel)).to(dec.device)
        for i in range(batch_size):
            tmp_label = label[i].cpu().numpy()
            feature_rec_tokens[:, i, :] = self.output_proj_list[tmp_label](dec[:, i, :])

        rec_feature = rearrange(
            feature_rec_tokens, "(h w) b c -> b c h w", h=self.feature_size[0]
        )

        pred = torch.sqrt(
            torch.sum((rec_feature - org_feature) ** 2, dim=1, keepdim=True)
        )
        pred = self.upsample(pred)

        # loss
        feature_loss = self.feature_loss(rec_feature, org_feature)
        latent_loss = diff.mean()
        loss = self.latent_loss_weight * latent_loss + feature_loss

        output = {
            'feature_rec': rec_feature,
            'feature_align': org_feature,
            'pred': pred,
            'pred_imgs': input,
            'loss': loss,
            'feature_loss': feature_loss,
            'latent_loss': latent_loss
        }
        inputs.update(output)

        return inputs

    def extract_feature(self, input):
        enc = self.enc(input)
        # enc_t = enc['features'][-1]
        feature_list = []
        for i in range(len(enc['features'])):
            feature_list.append(self.upsample_list[i](enc['features'][i]))
        enc_t = torch.cat(feature_list, dim=1)

        return enc_t

    def encode(self, input_list, label):
        quant_list = []
        id_list = []
        # quantize enc_4
        quant_4 = input_list[-1]

        new_quant_4 = torch.zeros_like(quant_4).to(quant_4.device)
        new_id_4 = torch.zeros(size=quant_4.size()[:-1]).to(quant_4.device)
        new_diff_4 = torch.zeros(size=(1,)).to(quant_4.device)
        for q_i in range(quant_4.size()[0]):
            tmp_label = label[q_i].cpu().numpy()
            # print(tmp_label)
            tmp_quant_4, tmp_diff_4, tmp_id_4 = self.quantize_list_5[tmp_label](quant_4[q_i])
            new_quant_4[q_i,:,:] = tmp_quant_4
            new_diff_4 += tmp_diff_4
            new_id_4[q_i, :] = tmp_id_4

        # quantize (enc_3, quant_4)
        quant_43 = torch.cat([input_list[-2], new_quant_4], dim=2)
        new_quant_43 = torch.zeros_like(quant_43).to(quant_43.device)
        new_id_43 = torch.zeros(size=quant_43.size()[:-1]).to(quant_43.device)
        new_diff_43 = torch.zeros(size=(1,)).to(quant_43.device)
        for q_i in range(quant_43.size()[0]):
            tmp_label = label[q_i].cpu().numpy()
            tmp_quant_43, tmp_diff_43, tmp_id_43 = self.quantize_list_4[tmp_label](quant_43[q_i])
            new_quant_43[q_i,:,:] = tmp_quant_43
            new_diff_43 += tmp_diff_43
            new_id_43[q_i, :] = tmp_id_43
        quant_list.append(new_quant_43)
        id_list.append(new_id_43)

        # quantize (enc_2, quant_4)
        quant_42 = torch.cat([input_list[-3], new_quant_4], dim=2)
        new_quant_42 = torch.zeros_like(quant_42).to(quant_42.device)
        new_id_42 = torch.zeros(size=quant_42.size()[:-1]).to(quant_42.device)
        new_diff_42 = torch.zeros(size=(1,)).to(quant_42.device)
        for q_i in range(quant_42.size()[0]):
            tmp_label = label[q_i].cpu().numpy()
            tmp_quant_42, tmp_diff_42, tmp_id_42 = self.quantize_list_3[tmp_label](quant_42[q_i])
            new_quant_42[q_i,:,:] = tmp_quant_42
            new_diff_42 += tmp_diff_42
            new_id_42[q_i, :] = tmp_id_42
        quant_list.append(new_quant_42)
        id_list.append(new_id_42)

        # quantize (enc_1, quant_4)
        quant_41 = torch.cat([input_list[-4], new_quant_4], dim=2)
        new_quant_41 = torch.zeros_like(quant_41).to(quant_41.device)
        new_id_41 = torch.zeros(size=quant_41.size()[:-1]).to(quant_41.device)
        new_diff_41 = torch.zeros(size=(1,)).to(quant_41.device)
        for q_i in range(quant_41.size()[0]):
            tmp_label = label[q_i].cpu().numpy()
            tmp_quant_41, tmp_diff_41, tmp_id_41 = self.quantize_list_2[tmp_label](quant_41[q_i])
            new_quant_41[q_i,:,:] = tmp_quant_41
            new_diff_41 += tmp_diff_41
            new_id_41[q_i,:] = tmp_id_41
        quant_list.append(new_quant_41)
        id_list.append(new_id_41)

        # quantize (enc_0, quant_4)
        quant_40 = torch.cat([input_list[-5], new_quant_4], dim=2)
        new_quant_40 = torch.zeros_like(quant_40).to(quant_40.device)
        new_id_40 = torch.zeros(size=quant_40.size()[:-1]).to(quant_40.device)
        new_diff_40 = torch.zeros(size=(1,)).to(quant_40.device)
        for q_i in range(quant_40.size()[0]):
            tmp_label = label[q_i].cpu().numpy()
            tmp_quant_40, tmp_diff_40, tmp_id_40 = self.quantize_list_1[tmp_label](quant_40[q_i])
            new_quant_40[q_i,:,:] = tmp_quant_40
            new_diff_40 += tmp_diff_40
            new_id_40[q_i,:] = tmp_id_40
        quant_list.append(new_quant_40)
        id_list.append(new_id_40)

        new_diff_4 = new_diff_4 / quant_4.size()[0]
        new_diff_43 = new_diff_43 / quant_43.size()[0]
        new_diff_42 = new_diff_42 / quant_42.size()[0]
        new_diff_41 = new_diff_41 / quant_41.size()[0]
        new_diff_40 = new_diff_40 / quant_40.size()[0]

        new_diff_4 = new_diff_4.unsqueeze(0)
        new_diff_43 = new_diff_43.unsqueeze(0)
        new_diff_42 = new_diff_42.unsqueeze(0)
        new_diff_41 = new_diff_41.unsqueeze(0)
        new_diff_40 = new_diff_40.unsqueeze(0)

        diff = new_diff_4 + new_diff_43 + new_diff_42 + new_diff_41 + new_diff_40

        return torch.stack(quant_list), diff, torch.stack(id_list)


