# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from .sam2_utils import LayerNorm2d, MLP
from .transformer import Attention
import torch.nn.functional as F

class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        activation: Type[nn.Module] = nn.GELU,
        num_multimask_outputs: int = 3,
        iou_head_depth: int = 3,
        # iou_head_hidden_dim: int = 256,
        # iou_prediction_use_sigmoid=False,
        # pred_obj_scores: bool = False,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.sparse_embedding = nn.Embedding(1, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        # self.iou_prediction_head = MLP(
        #     transformer_dim,
        #     iou_head_hidden_dim,
        #     self.num_mask_tokens,
        #     iou_head_depth,
        #     sigmoid_output=iou_prediction_use_sigmoid,
        # )


    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        multimask_output: bool,
        dense_prompt_embeddings=None,
        pre_output_tokens=None,
        sparse_prompt=None,
    ) :
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
          torch.Tensor: batched SAM token for mask output
        """
        masks, sparse_prompt, sam_tokens_out = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            dense_prompt_embeddings = dense_prompt_embeddings,
            pre_output_tokens = pre_output_tokens,
            sparse_prompt = sparse_prompt,
        )

        if multimask_output:
            masks = masks[:, 1:, :, :]
            # iou_pred = iou_pred[:, 1:]
            # max_iou_indices = torch.argmax(iou_pred, dim=1)
            # best_masks = masks[torch.arange(masks.shape[0]), max_iou_indices].unsqueeze(1)
            # masks = torch.cat([masks, best_masks], dim=1)
        else:
            masks = masks[:, 0:1, :, :]
            # iou_pred = iou_pred[:, 0:1]

        return masks, sam_tokens_out, sparse_prompt

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        dense_prompt_embeddings,
        pre_output_tokens,
        sparse_prompt,
    ) :
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        if pre_output_tokens is not None:
            output_tokens = torch.cat([self.iou_token.weight.unsqueeze(0).expand(image_embeddings.shape[0], -1, -1),
                                       pre_output_tokens], dim=1)
        else:
            mask_tokens = self.mask_tokens.weight
            output_tokens = torch.cat([self.iou_token.weight, mask_tokens], dim=0).unsqueeze(0).expand(
                image_embeddings.shape[0], -1, -1)

        if sparse_prompt is not None:
            sparse_prompts = torch.cat((sparse_prompt[0],sparse_prompt[1]), dim=1)
        else:
            sparse_prompts = self.sparse_embedding.weight.unsqueeze(0).expand(image_embeddings.shape[0], -1, -1)

        tokens = torch.cat([output_tokens, sparse_prompts], dim=1)

        src = image_embeddings
        b, c, h, w = src.shape
        if dense_prompt_embeddings is not None:
            src = dense_prompt_embeddings + src
        pos_src = image_pe

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        # iou_token_out = hs[:, 0, :]
        sam_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]
        sparse_prompt = hs[:, (1 + self.num_mask_tokens) : (2 + self.num_mask_tokens), :]

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](sam_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)

        # Upscale mask embeddings and predict masks using the mask tokens

        src = src.transpose(1, 2).view(b, c, h, w).contiguous()
        upscaled_embedding = self.output_upscaling(src)

        b, c, h, w = upscaled_embedding.shape#4,32,128,128

        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        # iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, sparse_prompt, sam_tokens_out

