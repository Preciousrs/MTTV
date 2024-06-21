#!/usr/bin/env python3

import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel, BertConfig
import copy
from torch.nn.parameter import Parameter

from mttv.models.image import MixImageEncoder
from mttv.models.image import RegionVisualFeatureEncoder


class ImageBertEmbeddings(nn.Module):
    def __init__(self, args, embeddings):
        super(ImageBertEmbeddings, self).__init__()
        self.args = args
        self.position_embeddings = embeddings.position_embeddings
        self.token_type_embeddings = copy.deepcopy(embeddings.token_type_embeddings)
        self.word_embeddings = embeddings.word_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, input_imgs, token_type_ids):
        bsz = input_imgs.size(0)
        seq_length = self.args.num_image_embeds + 2  # +2 for CLS and SEP Token

        cls_id = torch.LongTensor([self.args.vocab.stoi["[CLS]"]]).cuda()
        cls_id = cls_id.unsqueeze(0).expand(bsz, 1)
        cls_token_embeds = self.word_embeddings(cls_id)

        sep_id = torch.LongTensor([self.args.vocab.stoi["[SEP]"]]).cuda()
        sep_id = sep_id.unsqueeze(0).expand(bsz, 1)
        sep_token_embeds = self.word_embeddings(sep_id)
        
        imgs_embeddings = input_imgs
        token_embeddings = torch.cat(
            [cls_token_embeds, imgs_embeddings, sep_token_embeds], dim=1
        )
        

        position_ids = torch.arange(seq_length, dtype=torch.long).cuda()
        position_ids = position_ids.unsqueeze(0).expand(bsz, seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultimodalBertEncoder(nn.Module):
    def __init__(self, args):
        super(MultimodalBertEncoder, self).__init__()
        self.args = args
        bert = BertModel.from_pretrained(args.bert_model)
        self.txt_embeddings = bert.embeddings
        self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)

        # MixImageEncoder encode both global and entity-level visual feature
        self.img_encoder = MixImageEncoder(args)
        #region
        # self.img_encoder = RegionVisualFeatureEncoder(args)

        # set the super parameter N -> the number of LSTM layers
        self.lstm_layers = args.encoder_layer_num
        self.hidden_size = bert.config.hidden_size
        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, 
                            num_layers=self.lstm_layers, batch_first=True, bidirectional=True)
        self.pooler = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, input_txt, attention_mask, segment, global_image, region_image):
        bsz = input_txt.size(0)
        attention_mask = torch.cat(
            [
                torch.ones(bsz, self.args.num_image_embeds + 2).long().cuda(),
                attention_mask,
            ],
            dim=1,
        )
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        img_tok = (
            torch.cat([
                torch.LongTensor(input_txt.size(0), self.args.global_image_embeds + 1).fill_(0),
                torch.LongTensor(input_txt.size(0), self.args.region_image_embeds + 1).fill_(1)
            ], dim=1).cuda()
        )

        img = self.img_encoder(global_image, region_image)
        img_embed_out = self.img_embeddings(img, img_tok)
        txt_embed_out = self.txt_embeddings(input_txt, segment)
        encoder_input = torch.cat([img_embed_out, txt_embed_out], 1)  # Bx(TEXT+IMG)xHID

        # Pass through LSTM layers
        lstm_out, _ = self.lstm(encoder_input)
        
       # Use the pooler to get the final output
        pooled_output = self.pooler(lstm_out[:, 0])  # Use the output corresponding to the [CLS] token
        pooled_output = self.activation(pooled_output)

        return pooled_output


class MTTV(nn.Module):
    def __init__(self, args):
        super(MTTV, self).__init__()
        self.args = args
        self.enc = MultimodalBertEncoder(args)
        self.clf = nn.Linear(args.hidden_sz, args.n_classes, bias=False)

    def forward(self, txt, mask, segment, img, regions):
        x = self.enc(txt, mask, segment, img, regions)

        x = self.clf(x)

        return x


class MTTV_WithScalableClassifier(nn.Module):
    def __init__(self, mttv_model, scaling_factors):
        super(MTTV_WithScalableClassifier, self).__init__()
        self.enc = copy.deepcopy(mttv_model.enc)
        self.clf = copy.deepcopy(mttv_model.clf)
        self.scaling_factors = Parameter(torch.Tensor(scaling_factors))
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, txt, mask, segment, img, regions):
        x = self.enc(txt, mask, segment, img, regions)
        x = self.clf(x)
        x *= self.scaling_factors
        x = self.softmax(x)
        return x
