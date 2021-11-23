import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .backbone import ConvEmbeddingGC
from .transformer import DecoderLayer, Embeddings, PositionalEncoding,\
    MultiHeadAttention, PositionwiseFeedForward
from utils.convertor import BaseConvertor, TableMasterConvertor


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TableMasterConcat(nn.Module):
    def __init__(self, num_layer, self_attn, src_attn, feed_forward, dropout, d_model,
                 num_classes, start_idx=1, padding_idx=0, max_seq_len=500):
        super().__init__()
        self.num_layer = num_layer
        self.layers = clones(DecoderLayer(d_model, self_attn, src_attn, feed_forward, dropout), num_layer-1)
        self.cls_layer = DecoderLayer(d_model, self_attn, src_attn, feed_forward, dropout)
        self.bbox_layer = DecoderLayer(d_model, self_attn, src_attn, feed_forward, dropout)
        self.cls_fc = nn.Linear(d_model, num_classes)
        self.bbox_fc = nn.Sequential(nn.Linear(d_model, 4),
                                     nn.Sigmoid())
        self.norm = nn.LayerNorm(d_model)
        self.embedding = Embeddings(d_model, num_classes)
        self.positional_encoding = PositionalEncoding(d_model)
        self.SOS = start_idx
        self.PAD = padding_idx
        self.max_length = max_seq_len

    def make_mask(self, src, tgt):
        """
        src: [B, seq_len, d_model]
        tgt: [B, num_class]
        """
        tgt_pad_mask = (tgt != self.PAD).unsqueeze(1).unsqueeze(3).byte()  # B, 1, tgt_len, 1
        tgt_len = tgt.size(1)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.uint8, device=src.device))
        tgt_mask = tgt_pad_mask & tgt_sub_mask  # B, 1, tgt_len, tgt_len
        return None, tgt_mask

    def decode(self, tgt, feature, src_mask, tgt_mask):
        x = self.embedding(tgt)
        x = self.positional_encoding(x)
        for i, layer in enumerate(self.num_layer):
            x = layer(x, feature, src_mask, tgt_mask)

        cls_x = self.layer(x, feature, src_mask, tgt_mask)
        cls_x = self.norm(cls_x)

        bbox_x = self.layer(x, feature, src_mask, tgt_mask)
        bbox_x = self.norm(bbox_x)
        return self.cls_fc(cls_x), self.bbox_fc(bbox_x)

    def forward_train(self, feature, out_enc, tgt_dict):
        # feat is feature after backbone before fe
        # out_enc is feature after pe.
        device = feature.device
        padded_tgt = tgt_dict.to(device)
        src_mask, tgt_mask = self.make_mask(out_enc, padded_tgt[:, :-1])
        return self.decode(padded_tgt[:, :-1], out_enc, src_mask, tgt_mask)

    def greedy_forward(self, SOS, feature, src_mask):
        input = SOS
        output = None
        for idx in range(self.max_length + 1):
            _, tgt_mask = self.make_mask(feature, input)
            out, bbox_output = self.decode(input, feature, src_mask, tgt_mask)
            output = out
            prob = F.softmax(out, dim=-1)
            _, next_word = torch.max(prob, dim=-1)
            input = torch.cat([input, next_word[:, -1].unsqueeze(-1)], dim=-1)
        return output, bbox_output

    def forward_test(self, feature, out_enc):
        batch_size = out_enc.shape[0]
        SOS = torch.zeros(batch_size).long().to(out_enc.device)
        SOS[:] = self.SOS
        SOS = SOS.unsqueeze(1)
        output, bbox_output = self.greedy_forward(SOS, out_enc, src_mask=None)
        return output, bbox_output

    def forward(self, feat, out_enc, tgt_dict=None, train_mode=True):
        if train_mode:
            return self.forward_train(feat, out_enc, tgt_dict)
        return self.forward_test(feat, out_enc)


class TableMASTER(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        common_kwargs = kwargs['common_kwargs']
        backbone_kwargs = kwargs['backbone_kwargs']
        encoder_kwargs = kwargs['encoder_kwargs']
        decoder_kwargs = kwargs['decoder_kwargs']
        self.with_encoder = common_kwargs['with_encoder']
        self.build_model(common_kwargs, backbone_kwargs, encoder_kwargs, decoder_kwargs)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def build_model(self, common_kwargs, backbone_kwargs, encoder_kwargs, decoder_kwargs):
        common_kwargs['n_class'] = BaseConvertor(dict_file='utils/structure_alphabet.txt').n_classes
        tgt_vocab = common_kwargs['n_class']
        # shared
        d_model = common_kwargs['model_size']
        h = common_kwargs['multiheads']

        # encoder cfg
        encoder_dropout = encoder_kwargs['dropout']
        encoder_position = PositionalEncoding(d_model, encoder_dropout)

        # decoder cfg
        decoders = decoder_kwargs['stacks']
        decoder_dropout = decoder_kwargs['dropout']
        decoder_d_ff = decoder_kwargs['feed_forward_size']

        decoder_attn = MultiHeadAttention(h, d_model, decoder_dropout)
        decoder_ff = PositionwiseFeedForward(d_model, decoder_d_ff, decoder_dropout)
        decoder_position = PositionalEncoding(d_model, decoder_dropout)

        conv_embedding_gc = ConvEmbeddingGC(**backbone_kwargs)
        decoder = TableMasterConcat(num_layer=decoders, self_attn=copy.deepcopy(decoder_attn),
                                    src_attn=copy.deepcopy(decoder_attn), feed_forward=copy.deepcopy(decoder_ff),
                                    dropout=decoder_dropout, d_model=d_model, num_classes=tgt_vocab)

        src_embed = nn.Sequential(conv_embedding_gc, copy.deepcopy(encoder_position))
        tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), copy.deepcopy(decoder_position))
        generator = Generator(d_model, tgt_vocab)

        self.encoder = None
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src):
        return self.src_embed(src)

    def forward(self, *input):
        src = input[0]
        tgt = input[1]  # (b, len_tgt) target or query, input of decoder

        # output = self.decode(src, src_mask, tgt, tgt_mask)
        output = self.decoder(self.encode(src), None)
        return self.generator(output)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def model_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params


class Generator(nn.Module):
    """
    Define standard linear + softmax generation step.
    """

    def __init__(self, hidden_dim, vocab_size):
        """

        :param hidden_dim: dim of model
        :param vocab_size: size of vocabulary
        """
        super(Generator, self).__init__()

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, *input):
        x = input[0]
        return self.fc(x)