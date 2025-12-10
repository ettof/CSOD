import torch
from torch import nn
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .image_encoder import FpnNeck
from .hieradet import Hiera
from .position_encoding import PositionEmbeddingSine
from .transformer import TwoWayTransformer

from typing import Any, Optional, Tuple, Type

from reprlib import recursive_repr
import numpy as np
from torch.nn import functional as F
from .sam2_utils import LayerNorm2d, MLP

class CSSAM(nn.Module):
    def __init__(self, model_size):
        super().__init__()

        self.img_size = 512
        self.model_size = model_size
        self.scalp = 1

        self.position_encoding = PositionEmbeddingSine(
            num_pos_feats=256,
            normalize=True,
            scale=None,
            temperature=10000
        )

        # large_config#############################################
        if self.model_size == "large":

            self.trunk = Hiera(
                embed_dim=144,
                num_heads=2,
                stages=[2, 6, 36, 4],
                global_att_blocks=[23, 33, 43],
                window_pos_embed_bkg_spatial_size=[7, 7],
                window_spec=[8, 4, 16, 8],
            )

            self.neck = FpnNeck(
                position_encoding=self.position_encoding,
                d_model=256,
                backbone_channel_list=[1152, 576, 288, 144],
                fpn_top_down_levels=[2, 3],
                fpn_interp_model="nearest"
            )

        # tiny config#############################################
        if self.model_size == "tiny":
            self.trunk = Hiera(
                    embed_dim=96,
                    num_heads=1,
                    stages=[1, 2, 7, 2],
                    global_att_blocks=[5, 7, 9],
                    window_pos_embed_bkg_spatial_size=[7, 7],
                    #window_spec=[8, 4, 16, 8]
                )
            self.neck = FpnNeck(
                    position_encoding=self.position_encoding,
                    d_model=256,
                    backbone_channel_list=[768, 384, 192, 96],
                    fpn_top_down_levels=[2, 3],  # 输出第 2 和第 3 层
                    fpn_interp_model="nearest"
                )

        # base_plus_config#############################################
        if self.model_size == "base":
            self.trunk = Hiera(
                embed_dim=112,
                num_heads=2,
            )

            self.neck = FpnNeck(
                position_encoding=self.position_encoding,
                d_model=256,
                backbone_channel_list=[896, 448, 224, 112],
                fpn_top_down_levels=[2, 3],  # 输出第 2 和第 3 层
                fpn_interp_model="nearest"
            )

        self.PPG_mask_decoder = MaskDecoder(
            transformer=TwoWayTransformer(
                depth = 2,
                embedding_dim = 256,
                mlp_dim = 2048,
                num_heads = 8
            ),
            transformer_dim = 256,
        )

        self.mask_decoder = MaskDecoder(
            transformer=TwoWayTransformer(
                depth = 2,
                embedding_dim = 256,
                mlp_dim = 2048,
                num_heads = 8
            ),
            transformer_dim = 256,
        )

        self.NPG_mask_decoder = MaskDecoder(
            transformer=TwoWayTransformer(
                depth = 2,
                embedding_dim = 256,
                mlp_dim = 2048,
                num_heads = 8
            ),
            transformer_dim = 256,
        )

        # build PromptEncoder and MaskDecoder from SAM
        # (their hyperparameters like `mask_in_chans=16` are from SAM code)
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(32,32),
            input_image_size=(128, 128),
            mask_in_chans=16,
        )

        self.pim = PIM(256)

    def forward(self, img):

        sam_img = F.interpolate(img, [self.img_size,self.img_size], mode = 'bilinear')


        features, pos, class_predict = self.neck(self.trunk(sam_img))


        src = features[-1]
        features, pos = features[: -self.scalp], pos[: -self.scalp]
        encode_out = {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }

        #extract intermediate outputs for deep supervision to prevent model overfitting on the detail enhancement module.
        img_pe = encode_out["vision_pos_enc"][-1]

        positive_dense_prompt, output_tokens, positive_sparse_prompt = self.PPG_mask_decoder(
            encode_out["backbone_fpn"][-1],
            img_pe,
            False,
        )

        negative_dense_prompt, _, negative_sparse_prompt = self.NPG_mask_decoder(
            encode_out["backbone_fpn"][-1],
            img_pe,
            False,
        )

        positive_dense_prompt1 = torch.nn.functional.interpolate(positive_dense_prompt,[128,128], mode = 'bilinear', align_corners = False)
        negative_dense_prompt1 = torch.nn.functional.interpolate(negative_dense_prompt, [128, 128], mode='bilinear', align_corners=False)

        _, dense_embeddings = self.sam_prompt_encoder(
            points=None,
            boxes=None,
            masks=positive_dense_prompt1,
        )
        _, neg_dense_embeddings = self.sam_prompt_encoder(
            points=None,
            boxes=None,
            masks=negative_dense_prompt1,
        )

        positive_sparse_token = self.sam_prompt_encoder.point_embeddings[1].weight + positive_sparse_prompt
        negative_sparse_token = self.sam_prompt_encoder.point_embeddings[0].weight + negative_sparse_prompt
        dense_prompt_embeddings = self.pim(dense_embeddings, neg_dense_embeddings)

        masks,_,_ = self.mask_decoder(
            encode_out["backbone_fpn"][-1],
            img_pe,
            False,
            dense_prompt_embeddings=dense_prompt_embeddings,
            pre_output_tokens=output_tokens,
            sparse_prompt=[positive_sparse_token,negative_sparse_token],
        )
        negative_dense_prompt = torch.nn.functional.interpolate(negative_dense_prompt, [self.img_size, self.img_size], mode='bilinear',align_corners=False)
        masks = torch.nn.functional.interpolate(masks,[self.img_size,self.img_size], mode = 'bilinear', align_corners = False)
        positive_dense_prompt = torch.nn.functional.interpolate(positive_dense_prompt, [self.img_size,self.img_size], mode='bilinear', align_corners=False)

        return positive_dense_prompt, masks, negative_dense_prompt, class_predict

class PIM(nn.Module):
    def __init__(self, indim,activation: Type[nn.Module] = nn.GELU):
        super(PIM, self).__init__()
        self.norm = LayerNorm2d(indim)
        self.conv = nn.Sequential(
            nn.Conv2d(indim*2, indim*2, kernel_size=3, padding=1),
            activation(),
            nn.Conv2d(indim*2, indim, kernel_size=1),
        )

    def forward(self,d_pos,d_neg):
        manhattan_distance = torch.abs(d_pos - d_neg)
        reverse_sim = 1 - self.norm(manhattan_distance)
        # reverse_sim = reverse_sim / max_distance
        src1_r = reverse_sim * d_pos
        src2_r = reverse_sim * d_neg
        x = self.conv(torch.cat((src1_r, src2_r), dim=1))
        x = d_pos + x

        return x