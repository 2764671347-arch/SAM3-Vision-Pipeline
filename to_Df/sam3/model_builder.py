# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

from __future__ import annotations

import json
import os
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from iopath.common.file_io import g_pathmgr
from sam3.model.decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerDecoderLayerv2,
    TransformerEncoderCrossAttention,
)
from sam3.model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from sam3.model.geometry_encoders import SequenceGeometryEncoder
from sam3.model.maskformer_segmentation import PixelDecoder, UniversalSegmentationHead
from sam3.model.memory import (
    CXBlock,
    SimpleFuser,
    SimpleMaskDownSampler,
    SimpleMaskEncoder,
)
from sam3.model.model_misc import (
    DotProductScoring,
    MLP,
    MultiheadAttentionWrapper as MultiheadAttention,
    TransformerWrapper,
)
from sam3.model.necks import Sam3DualViTDetNeck
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.model.sam3_image import Sam3Image
from sam3.model.text_encoder_ve import VETextEncoder
from sam3.model.tokenizer_ve import SimpleTokenizer
from sam3.model.vitdet import ViT
from sam3.model.vl_combiner import SAM3VLBackbone
from sam3.sam.transformer import RoPEAttention


# Setup TensorFloat-32 for Ampere GPUs if available
def _setup_tf32() -> None:
    """Enable TensorFloat-32 for Ampere GPUs if available."""
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        if device_props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


_setup_tf32()


DEFAULT_IMAGE_MODEL_BUILD_CONFIG = {
    "image_size": 1008,
    "patch_size": 14,
    "pretrain_img_size": 336,
    "vit_embed_dim": 1024,
    "vit_depth": 32,
    "vit_num_heads": 16,
    "vit_mlp_ratio": 4.625,
    "vit_drop_path_rate": 0.1,
    "vit_global_att_blocks": (7, 15, 23, 31),
    "vit_window_size": 24,
    "d_model": 256,
    "transformer_heads": 8,
    "encoder_layers": 6,
    "decoder_layers": 6,
    "transformer_ffn_dim": 2048,
    "encoder_dropout": 0.1,
    "decoder_dropout": 0.1,
    "num_queries": 200,
    "geo_encoder_layers": 3,
    "text_width": 1024,
    "text_layers": 24,
    "text_heads": 16,
    "prompt_mlp_hidden_dim": 2048,
    "segmentation_upsampling_stages": 3,
    "freeze_image_tower": False,
    "freeze_text_tower": False,
    "freeze_transformer": False,
    "freeze_prompt_scorer": False,
    "freeze_segmentation_head": False,
    "freeze_geometry_encoder": False,
}


def _normalize_blocks(blocks: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(block) for block in blocks)


def _freeze_module(module: Optional[nn.Module]) -> None:
    if module is None:
        return
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False


def _read_image_model_build_config(model_config_path: Optional[str]) -> dict[str, Any]:
    if model_config_path is None:
        return {}
    with g_pathmgr.open(model_config_path, "r") as fopen:
        config = json.load(fopen)
    if not isinstance(config, dict):
        raise ValueError("model build config must be a JSON object")
    if "build_args" in config:
        config = config["build_args"]
    if not isinstance(config, dict):
        raise ValueError("model build config build_args must be a JSON object")
    return config


def _create_position_encoding(
    d_model: int = 256, precompute_resolution: Optional[int] = None
):
    """Create position encoding for visual backbone."""
    return PositionEmbeddingSine(
        num_pos_feats=d_model,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
    )


def _create_vit_backbone(
    image_size: int = 1008,
    patch_size: int = 14,
    pretrain_img_size: int = 336,
    embed_dim: int = 1024,
    depth: int = 32,
    num_heads: int = 16,
    mlp_ratio: float = 4.625,
    drop_path_rate: float = 0.1,
    global_att_blocks: Sequence[int] = (7, 15, 23, 31),
    window_size: int = 24,
    compile_mode=None,
):
    """Create ViT backbone for visual feature extraction."""
    return ViT(
        img_size=image_size,
        pretrain_img_size=pretrain_img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        norm_layer="LayerNorm",
        drop_path_rate=drop_path_rate,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=_normalize_blocks(global_att_blocks),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=window_size,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=compile_mode,
    )


def _create_vit_neck(
    position_encoding,
    vit_backbone,
    d_model: int = 256,
    enable_inst_interactivity: bool = False,
):
    """Create ViT neck for feature pyramid."""
    return Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=d_model,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit_backbone,
        add_sam2_neck=enable_inst_interactivity,
    )


def _create_vl_backbone(vit_neck, text_encoder):
    """Create visual-language backbone."""
    return SAM3VLBackbone(visual=vit_neck, text=text_encoder, scalp=1)


def _create_transformer_encoder(
    d_model: int = 256,
    transformer_heads: int = 8,
    encoder_layers: int = 6,
    transformer_ffn_dim: int = 2048,
    encoder_dropout: float = 0.1,
    use_act_checkpoint: bool = True,
) -> TransformerEncoderFusion:
    """Create transformer encoder with its layer."""
    encoder_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=d_model,
        dim_feedforward=transformer_ffn_dim,
        dropout=encoder_dropout,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=transformer_heads,
            dropout=encoder_dropout,
            embed_dim=d_model,
            batch_first=True,
        ),
        cross_attention=MultiheadAttention(
            num_heads=transformer_heads,
            dropout=encoder_dropout,
            embed_dim=d_model,
            batch_first=True,
        ),
    )

    encoder = TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=encoder_layers,
        d_model=d_model,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=use_act_checkpoint,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )
    return encoder


def _create_transformer_decoder(
    d_model: int = 256,
    transformer_heads: int = 8,
    decoder_layers: int = 6,
    num_queries: int = 200,
    transformer_ffn_dim: int = 2048,
    decoder_dropout: float = 0.1,
    image_size: int = 1008,
    patch_size: int = 14,
    use_act_checkpoint: bool = True,
    has_presence_token: bool = True,
) -> TransformerDecoder:
    """Create transformer decoder with its layer."""
    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=d_model,
        dim_feedforward=transformer_ffn_dim,
        dropout=decoder_dropout,
        cross_attention=MultiheadAttention(
            num_heads=transformer_heads,
            dropout=decoder_dropout,
            embed_dim=d_model,
        ),
        n_heads=transformer_heads,
        use_text_cross_attention=True,
    )

    decoder = TransformerDecoder(
        layer=decoder_layer,
        num_layers=decoder_layers,
        num_queries=num_queries,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=d_model,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        resolution=image_size,
        stride=patch_size,
        use_act_checkpoint=use_act_checkpoint,
        presence_token=has_presence_token,
    )
    return decoder


def _create_dot_product_scoring(
    d_model: int = 256,
    prompt_mlp_hidden_dim: int = 2048,
    dropout: float = 0.1,
):
    """Create dot product scoring module."""
    prompt_mlp = MLP(
        input_dim=d_model,
        hidden_dim=prompt_mlp_hidden_dim,
        output_dim=d_model,
        num_layers=2,
        dropout=dropout,
        residual=True,
        out_norm=nn.LayerNorm(d_model),
    )
    return DotProductScoring(d_model=d_model, d_proj=d_model, prompt_mlp=prompt_mlp)


def _create_segmentation_head(
    d_model: int = 256,
    upsampling_stages: int = 3,
    transformer_heads: int = 8,
    compile_mode=None,
    use_act_checkpoint: bool = True,
):
    """Create segmentation head with pixel decoder."""
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=upsampling_stages,
        interpolation_mode="nearest",
        hidden_dim=d_model,
        compile_mode=compile_mode,
    )

    cross_attend_prompt = MultiheadAttention(
        num_heads=transformer_heads,
        dropout=0,
        embed_dim=d_model,
    )

    segmentation_head = UniversalSegmentationHead(
        hidden_dim=d_model,
        upsampling_stages=upsampling_stages,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        act_ckpt=use_act_checkpoint,
        cross_attend_prompt=cross_attend_prompt,
        pixel_decoder=pixel_decoder,
    )
    return segmentation_head


def _create_geometry_encoder(
    d_model: int = 256,
    transformer_heads: int = 8,
    transformer_ffn_dim: int = 2048,
    encoder_dropout: float = 0.1,
    geo_encoder_layers: int = 3,
    use_act_checkpoint: bool = True,
):
    """Create geometry encoder with all its components."""
    # Create position encoding for geometry encoder
    geo_pos_enc = _create_position_encoding(d_model=d_model)
    # Create CX block for fuser
    cx_block = CXBlock(
        dim=d_model,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )
    # Create geometry encoder layer
    geo_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=d_model,
        dim_feedforward=transformer_ffn_dim,
        dropout=encoder_dropout,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=transformer_heads,
            dropout=encoder_dropout,
            embed_dim=d_model,
            batch_first=False,
        ),
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        cross_attention=MultiheadAttention(
            num_heads=transformer_heads,
            dropout=encoder_dropout,
            embed_dim=d_model,
            batch_first=False,
        ),
    )

    # Create geometry encoder
    input_geometry_encoder = SequenceGeometryEncoder(
        pos_enc=geo_pos_enc,
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=d_model,
        num_layers=geo_encoder_layers,
        layer=geo_layer,
        use_act_ckpt=use_act_checkpoint,
        add_cls=True,
        add_post_encode_proj=True,
    )
    return input_geometry_encoder


def _create_sam3_model(
    backbone,
    transformer,
    input_geometry_encoder,
    segmentation_head,
    dot_prod_scoring,
    inst_interactive_predictor,
    eval_mode,
):
    """Create the SAM3 image model."""
    common_params = {
        "backbone": backbone,
        "transformer": transformer,
        "input_geometry_encoder": input_geometry_encoder,
        "segmentation_head": segmentation_head,
        "num_feature_levels": 1,
        "o2m_mask_predict": True,
        "dot_prod_scoring": dot_prod_scoring,
        "use_instance_query": False,
        "multimask_output": True,
        "inst_interactive_predictor": inst_interactive_predictor,
    }

    matcher = None
    if not eval_mode:
        from sam3.train.matcher import BinaryHungarianMatcherV2

        matcher = BinaryHungarianMatcherV2(
            focal=True,
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            alpha=0.25,
            gamma=2,
            stable=False,
        )
    common_params["matcher"] = matcher
    model = Sam3Image(**common_params)

    return model


def _create_tracker_maskmem_backbone():
    """Create the SAM3 Tracker memory encoder."""
    # Position encoding for mask memory backbone
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=64,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=1008,
    )

    # Mask processing components
    mask_downsampler = SimpleMaskDownSampler(
        kernel_size=3, stride=2, padding=1, interpol_size=[1152, 1152]
    )

    cx_block_layer = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )

    fuser = SimpleFuser(layer=cx_block_layer, num_layers=2)

    maskmem_backbone = SimpleMaskEncoder(
        out_dim=64,
        position_encoding=position_encoding,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
    )

    return maskmem_backbone


def _create_tracker_transformer():
    """Create the SAM3 Tracker transformer components."""
    # Self attention
    self_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        use_fa3=False,
        use_rope_real=False,
    )

    # Cross attention
    cross_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        kv_in_dim=64,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        rope_k_repeat=True,
        use_fa3=False,
        use_rope_real=False,
    )

    # Encoder layer
    encoder_layer = TransformerDecoderLayerv2(
        cross_attention_first=False,
        activation="relu",
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=self_attention,
        d_model=256,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        cross_attention=cross_attention,
    )

    # Encoder
    encoder = TransformerEncoderCrossAttention(
        remove_cross_attention_layers=[],
        batch_first=True,
        d_model=256,
        frozen=False,
        pos_enc_at_input=True,
        layer=encoder_layer,
        num_layers=4,
        use_act_checkpoint=False,
    )

    # Transformer wrapper
    transformer = TransformerWrapper(
        encoder=encoder,
        decoder=None,
        d_model=256,
    )

    return transformer


def build_tracker(
    apply_temporal_disambiguation: bool, with_backbone: bool = False, compile_mode=None
) -> Sam3TrackerPredictor:
    """
    Build the SAM3 Tracker module for video tracking.

    Returns:
        Sam3TrackerPredictor: Wrapped SAM3 Tracker module
    """
    from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor

    # Create model components
    maskmem_backbone = _create_tracker_maskmem_backbone()
    transformer = _create_tracker_transformer()
    backbone = None
    if with_backbone:
        vision_backbone = _create_vision_backbone(compile_mode=compile_mode)
        backbone = SAM3VLBackbone(scalp=1, visual=vision_backbone, text=None)
    # Create the Tracker module
    model = Sam3TrackerPredictor(
        image_size=1008,
        num_maskmem=7,
        backbone=backbone,
        backbone_stride=14,
        transformer=transformer,
        maskmem_backbone=maskmem_backbone,
        # SAM parameters
        multimask_output_in_sam=True,
        # Evaluation
        forward_backbone_per_frame_for_eval=True,
        trim_past_non_cond_mem_for_eval=False,
        # Multimask
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        # Additional settings
        always_start_from_first_ann_frame=False,
        # Mask overlap
        non_overlap_masks_for_mem_enc=False,
        non_overlap_masks_for_output=False,
        max_cond_frames_in_attn=4,
        offload_output_to_cpu_for_eval=False,
        # SAM decoder settings
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
        clear_non_cond_mem_around_input=True,
        fill_hole_area=0,
        use_memory_selection=apply_temporal_disambiguation,
    )

    return model


def _create_text_encoder(
    bpe_path: str,
    d_model: int = 256,
    text_width: int = 1024,
    text_layers: int = 24,
    text_heads: int = 16,
) -> VETextEncoder:
    """Create SAM3 text encoder."""
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    return VETextEncoder(
        tokenizer=tokenizer,
        d_model=d_model,
        width=text_width,
        heads=text_heads,
        layers=text_layers,
    )


def _create_vision_backbone(
    image_size: int = 1008,
    patch_size: int = 14,
    pretrain_img_size: int = 336,
    vit_embed_dim: int = 1024,
    vit_depth: int = 32,
    vit_num_heads: int = 16,
    vit_mlp_ratio: float = 4.625,
    vit_drop_path_rate: float = 0.1,
    vit_global_att_blocks: Sequence[int] = (7, 15, 23, 31),
    vit_window_size: int = 24,
    d_model: int = 256,
    compile_mode=None,
    enable_inst_interactivity: bool = True,
) -> Sam3DualViTDetNeck:
    """Create SAM3 visual backbone with ViT and neck."""
    # Position encoding
    position_encoding = _create_position_encoding(
        d_model=d_model, precompute_resolution=image_size
    )
    # ViT backbone
    vit_backbone: ViT = _create_vit_backbone(
        image_size=image_size,
        patch_size=patch_size,
        pretrain_img_size=pretrain_img_size,
        embed_dim=vit_embed_dim,
        depth=vit_depth,
        num_heads=vit_num_heads,
        mlp_ratio=vit_mlp_ratio,
        drop_path_rate=vit_drop_path_rate,
        global_att_blocks=vit_global_att_blocks,
        window_size=vit_window_size,
        compile_mode=compile_mode,
    )
    vit_neck: Sam3DualViTDetNeck = _create_vit_neck(
        position_encoding,
        vit_backbone,
        d_model=d_model,
        enable_inst_interactivity=enable_inst_interactivity,
    )
    # Visual neck
    return vit_neck


def _create_sam3_transformer(
    d_model: int = 256,
    transformer_heads: int = 8,
    encoder_layers: int = 6,
    decoder_layers: int = 6,
    transformer_ffn_dim: int = 2048,
    encoder_dropout: float = 0.1,
    decoder_dropout: float = 0.1,
    num_queries: int = 200,
    image_size: int = 1008,
    patch_size: int = 14,
    use_act_checkpoint: bool = True,
    has_presence_token: bool = True,
) -> TransformerWrapper:
    """Create SAM3 transformer encoder and decoder."""
    encoder: TransformerEncoderFusion = _create_transformer_encoder(
        d_model=d_model,
        transformer_heads=transformer_heads,
        encoder_layers=encoder_layers,
        transformer_ffn_dim=transformer_ffn_dim,
        encoder_dropout=encoder_dropout,
        use_act_checkpoint=use_act_checkpoint,
    )
    decoder: TransformerDecoder = _create_transformer_decoder(
        d_model=d_model,
        transformer_heads=transformer_heads,
        decoder_layers=decoder_layers,
        num_queries=num_queries,
        transformer_ffn_dim=transformer_ffn_dim,
        decoder_dropout=decoder_dropout,
        image_size=image_size,
        patch_size=patch_size,
        use_act_checkpoint=use_act_checkpoint,
        has_presence_token=has_presence_token,
    )

    return TransformerWrapper(encoder=encoder, decoder=decoder, d_model=d_model)


def _load_checkpoint(model, checkpoint_path):
    """Load model checkpoint from file."""
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    sam3_image_ckpt = {
        k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k
    }
    if model.inst_interactive_predictor is not None:
        sam3_image_ckpt.update(
            {
                k.replace("tracker.", "inst_interactive_predictor.model."): v
                for k, v in ckpt.items()
                if "tracker" in k
            }
        )
    missing_keys, _ = model.load_state_dict(sam3_image_ckpt, strict=False)
    if len(missing_keys) > 0:
        print(
            f"loaded {checkpoint_path} and found "
            f"missing and/or unexpected keys:\n{missing_keys=}"
        )


def _setup_device_and_mode(model, device, eval_mode):
    """Setup model device and evaluation mode."""
    if device == "cuda":
        model = model.cuda()
    if eval_mode:
        model.eval()
    return model


def build_sam3_image_model(
    bpe_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    eval_mode=True,
    checkpoint_path=None,
    load_from_HF=True,
    enable_segmentation=True,
    enable_inst_interactivity=False,
    model_config_path=None,
    compile=False,
    image_size=1008,
    patch_size=14,
    pretrain_img_size=336,
    vit_embed_dim=1024,
    vit_depth=32,
    vit_num_heads=16,
    vit_mlp_ratio=4.625,
    vit_drop_path_rate=0.1,
    vit_global_att_blocks=(7, 15, 23, 31),
    vit_window_size=24,
    d_model=256,
    transformer_heads=8,
    encoder_layers=6,
    decoder_layers=6,
    transformer_ffn_dim=2048,
    encoder_dropout=0.1,
    decoder_dropout=0.1,
    num_queries=200,
    geo_encoder_layers=3,
    text_width=1024,
    text_layers=24,
    text_heads=16,
    prompt_mlp_hidden_dim=2048,
    segmentation_upsampling_stages=3,
    freeze_image_tower=False,
    freeze_text_tower=False,
    freeze_transformer=False,
    freeze_prompt_scorer=False,
    freeze_segmentation_head=False,
    freeze_geometry_encoder=False,
):
    """
    Build SAM3 image model

    Args:
        bpe_path: Path to the BPE tokenizer vocabulary
        device: Device to load the model on ('cuda' or 'cpu')
        eval_mode: Whether to set the model to evaluation mode
        checkpoint_path: Optional path to model checkpoint
        enable_segmentation: Whether to enable segmentation head
        enable_inst_interactivity: Whether to enable instance interactivity (SAM 1 task)
        model_config_path: Optional JSON config file with build arguments
        compile_mode: To enable compilation, set to "default"

    Returns:
        A SAM3 image model
    """
    if bpe_path is None:
        bpe_path = os.path.join(
            os.path.dirname(__file__), "..", "assets", "bpe_simple_vocab_16e6.txt.gz"
        )
    if model_config_path is not None:
        config_overrides = _read_image_model_build_config(model_config_path)
        merged_kwargs = {
            "bpe_path": bpe_path,
            "device": device,
            "eval_mode": eval_mode,
            "checkpoint_path": checkpoint_path,
            "load_from_HF": load_from_HF,
            "enable_segmentation": enable_segmentation,
            "enable_inst_interactivity": enable_inst_interactivity,
            "compile": compile,
        }
        merged_kwargs.update(config_overrides)
        return build_sam3_image_model(**merged_kwargs)
    # Create visual components
    compile_mode = "default" if compile else None
    vision_encoder = _create_vision_backbone(
        image_size=image_size,
        patch_size=patch_size,
        pretrain_img_size=pretrain_img_size,
        vit_embed_dim=vit_embed_dim,
        vit_depth=vit_depth,
        vit_num_heads=vit_num_heads,
        vit_mlp_ratio=vit_mlp_ratio,
        vit_drop_path_rate=vit_drop_path_rate,
        vit_global_att_blocks=vit_global_att_blocks,
        vit_window_size=vit_window_size,
        d_model=d_model,
        compile_mode=compile_mode,
        enable_inst_interactivity=enable_inst_interactivity,
    )

    # Create text components
    text_encoder = _create_text_encoder(
        bpe_path=bpe_path,
        d_model=d_model,
        text_width=text_width,
        text_layers=text_layers,
        text_heads=text_heads,
    )

    # Create visual-language backbone
    backbone = _create_vl_backbone(vision_encoder, text_encoder)

    # Create transformer components
    transformer = _create_sam3_transformer(
        d_model=d_model,
        transformer_heads=transformer_heads,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        transformer_ffn_dim=transformer_ffn_dim,
        encoder_dropout=encoder_dropout,
        decoder_dropout=decoder_dropout,
        num_queries=num_queries,
        image_size=image_size,
        patch_size=patch_size,
    )

    # Create dot product scoring
    dot_prod_scoring = _create_dot_product_scoring(
        d_model=d_model,
        prompt_mlp_hidden_dim=prompt_mlp_hidden_dim,
        dropout=decoder_dropout,
    )

    # Create segmentation head if enabled
    segmentation_head = (
        _create_segmentation_head(
            d_model=d_model,
            upsampling_stages=segmentation_upsampling_stages,
            transformer_heads=transformer_heads,
            compile_mode=compile_mode,
        )
        if enable_segmentation
        else None
    )

    # Create geometry encoder
    input_geometry_encoder = _create_geometry_encoder(
        d_model=d_model,
        transformer_heads=transformer_heads,
        transformer_ffn_dim=transformer_ffn_dim,
        encoder_dropout=encoder_dropout,
        geo_encoder_layers=geo_encoder_layers,
    )
    if enable_inst_interactivity:
        from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor

        sam3_pvs_base = build_tracker(apply_temporal_disambiguation=False)
        inst_predictor = SAM3InteractiveImagePredictor(sam3_pvs_base)
    else:
        inst_predictor = None
    # Create the SAM3 model
    model = _create_sam3_model(
        backbone,
        transformer,
        input_geometry_encoder,
        segmentation_head,
        dot_prod_scoring,
        inst_predictor,
        eval_mode,
    )
    if freeze_image_tower:
        _freeze_module(backbone.vision_backbone)
    if freeze_text_tower:
        _freeze_module(backbone.language_backbone)
    if freeze_transformer:
        _freeze_module(transformer)
    if freeze_prompt_scorer:
        _freeze_module(dot_prod_scoring)
    if freeze_segmentation_head:
        _freeze_module(segmentation_head)
    if freeze_geometry_encoder:
        _freeze_module(input_geometry_encoder)

    uses_default_architecture = (
        image_size == DEFAULT_IMAGE_MODEL_BUILD_CONFIG["image_size"]
        and patch_size == DEFAULT_IMAGE_MODEL_BUILD_CONFIG["patch_size"]
        and pretrain_img_size == DEFAULT_IMAGE_MODEL_BUILD_CONFIG["pretrain_img_size"]
        and vit_embed_dim == DEFAULT_IMAGE_MODEL_BUILD_CONFIG["vit_embed_dim"]
        and vit_depth == DEFAULT_IMAGE_MODEL_BUILD_CONFIG["vit_depth"]
        and vit_num_heads == DEFAULT_IMAGE_MODEL_BUILD_CONFIG["vit_num_heads"]
        and vit_mlp_ratio == DEFAULT_IMAGE_MODEL_BUILD_CONFIG["vit_mlp_ratio"]
        and vit_drop_path_rate
        == DEFAULT_IMAGE_MODEL_BUILD_CONFIG["vit_drop_path_rate"]
        and _normalize_blocks(vit_global_att_blocks)
        == DEFAULT_IMAGE_MODEL_BUILD_CONFIG["vit_global_att_blocks"]
        and vit_window_size == DEFAULT_IMAGE_MODEL_BUILD_CONFIG["vit_window_size"]
        and d_model == DEFAULT_IMAGE_MODEL_BUILD_CONFIG["d_model"]
        and transformer_heads == DEFAULT_IMAGE_MODEL_BUILD_CONFIG["transformer_heads"]
        and encoder_layers == DEFAULT_IMAGE_MODEL_BUILD_CONFIG["encoder_layers"]
        and decoder_layers == DEFAULT_IMAGE_MODEL_BUILD_CONFIG["decoder_layers"]
        and transformer_ffn_dim
        == DEFAULT_IMAGE_MODEL_BUILD_CONFIG["transformer_ffn_dim"]
        and encoder_dropout == DEFAULT_IMAGE_MODEL_BUILD_CONFIG["encoder_dropout"]
        and decoder_dropout == DEFAULT_IMAGE_MODEL_BUILD_CONFIG["decoder_dropout"]
        and num_queries == DEFAULT_IMAGE_MODEL_BUILD_CONFIG["num_queries"]
        and geo_encoder_layers == DEFAULT_IMAGE_MODEL_BUILD_CONFIG["geo_encoder_layers"]
        and text_width == DEFAULT_IMAGE_MODEL_BUILD_CONFIG["text_width"]
        and text_layers == DEFAULT_IMAGE_MODEL_BUILD_CONFIG["text_layers"]
        and text_heads == DEFAULT_IMAGE_MODEL_BUILD_CONFIG["text_heads"]
        and prompt_mlp_hidden_dim
        == DEFAULT_IMAGE_MODEL_BUILD_CONFIG["prompt_mlp_hidden_dim"]
        and segmentation_upsampling_stages
        == DEFAULT_IMAGE_MODEL_BUILD_CONFIG["segmentation_upsampling_stages"]
    )
    if load_from_HF and checkpoint_path is None and not uses_default_architecture:
        raise ValueError(
            "Custom SAM3 image architectures require an explicit checkpoint_path or "
            "load_from_HF=False."
        )
    if load_from_HF and checkpoint_path is None:
        checkpoint_path = download_ckpt_from_hf()
    # Load checkpoint if provided
    if checkpoint_path is not None:
        _load_checkpoint(model, checkpoint_path)

    # Setup device and mode
    model = _setup_device_and_mode(model, device, eval_mode)

    return model


def build_sam3_image_model_from_config(
    model_config_path: str,
    **overrides,
):
    config = _read_image_model_build_config(model_config_path)
    config.update(overrides)
    return build_sam3_image_model(**config)


def download_ckpt_from_hf():
    SAM3_MODEL_ID = "facebook/sam3"
    SAM3_CKPT_NAME = "sam3.pt"
    SAM3_CFG_NAME = "config.json"
    _ = hf_hub_download(repo_id=SAM3_MODEL_ID, filename=SAM3_CFG_NAME)
    checkpoint_path = hf_hub_download(repo_id=SAM3_MODEL_ID, filename=SAM3_CKPT_NAME)
    return checkpoint_path


def build_sam3_video_model(
    checkpoint_path: Optional[str] = None,
    load_from_HF=True,
    bpe_path: Optional[str] = None,
    has_presence_token: bool = True,
    geo_encoder_use_img_cross_attn: bool = True,
    strict_state_dict_loading: bool = True,
    apply_temporal_disambiguation: bool = True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    compile=False,
) -> Sam3VideoInferenceWithInstanceInteractivity:
    """
    Build SAM3 dense tracking model.

    Args:
        checkpoint_path: Optional path to checkpoint file
        bpe_path: Path to the BPE tokenizer file

    Returns:
        Sam3VideoInferenceWithInstanceInteractivity: The instantiated dense tracking model
    """
    from sam3.model.sam3_image import Sam3ImageOnVideoMultiGPU
    from sam3.model.sam3_video_inference import (
        Sam3VideoInferenceWithInstanceInteractivity,
    )

    if bpe_path is None:
        bpe_path = os.path.join(
            os.path.dirname(__file__), "..", "assets", "bpe_simple_vocab_16e6.txt.gz"
        )

    # Build Tracker module
    tracker = build_tracker(apply_temporal_disambiguation=apply_temporal_disambiguation)

    # Build Detector components
    visual_neck = _create_vision_backbone()
    text_encoder = _create_text_encoder(bpe_path)
    backbone = SAM3VLBackbone(scalp=1, visual=visual_neck, text=text_encoder)
    transformer = _create_sam3_transformer(has_presence_token=has_presence_token)
    segmentation_head: UniversalSegmentationHead = _create_segmentation_head()
    input_geometry_encoder = _create_geometry_encoder()

    # Create main dot product scoring
    main_dot_prod_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    main_dot_prod_scoring = DotProductScoring(
        d_model=256, d_proj=256, prompt_mlp=main_dot_prod_mlp
    )

    # Build Detector module
    detector = Sam3ImageOnVideoMultiGPU(
        num_feature_levels=1,
        backbone=backbone,
        transformer=transformer,
        segmentation_head=segmentation_head,
        semantic_segmentation_head=None,
        input_geometry_encoder=input_geometry_encoder,
        use_early_fusion=True,
        use_dot_prod_scoring=True,
        dot_prod_scoring=main_dot_prod_scoring,
        supervise_joint_box_scores=has_presence_token,
    )

    # Build the main SAM3 video model
    if apply_temporal_disambiguation:
        model = Sam3VideoInferenceWithInstanceInteractivity(
            detector=detector,
            tracker=tracker,
            score_threshold_detection=0.5,
            assoc_iou_thresh=0.1,
            det_nms_thresh=0.1,
            new_det_thresh=0.7,
            hotstart_delay=15,
            hotstart_unmatch_thresh=8,
            hotstart_dup_thresh=8,
            suppress_unmatched_only_within_hotstart=True,
            min_trk_keep_alive=-1,
            max_trk_keep_alive=30,
            init_trk_keep_alive=30,
            suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
            suppress_det_close_to_boundary=False,
            fill_hole_area=16,
            recondition_every_nth_frame=16,
            masklet_confirmation_enable=False,
            decrease_trk_keep_alive_for_empty_masklets=False,
            image_size=1008,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            compile_model=compile,
        )
    else:
        # a version without any heuristics for ablation studies
        model = Sam3VideoInferenceWithInstanceInteractivity(
            detector=detector,
            tracker=tracker,
            score_threshold_detection=0.5,
            assoc_iou_thresh=0.1,
            det_nms_thresh=0.1,
            new_det_thresh=0.7,
            hotstart_delay=0,
            hotstart_unmatch_thresh=0,
            hotstart_dup_thresh=0,
            suppress_unmatched_only_within_hotstart=True,
            min_trk_keep_alive=-1,
            max_trk_keep_alive=30,
            init_trk_keep_alive=30,
            suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
            suppress_det_close_to_boundary=False,
            fill_hole_area=16,
            recondition_every_nth_frame=0,
            masklet_confirmation_enable=False,
            decrease_trk_keep_alive_for_empty_masklets=False,
            image_size=1008,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            compile_model=compile,
        )

    # Load checkpoint if provided
    if load_from_HF and checkpoint_path is None:
        checkpoint_path = download_ckpt_from_hf()
    if checkpoint_path is not None:
        with g_pathmgr.open(checkpoint_path, "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=True)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]

        missing_keys, unexpected_keys = model.load_state_dict(
            ckpt, strict=strict_state_dict_loading
        )
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

    model.to(device=device)
    return model


def build_sam3_video_predictor(*model_args, gpus_to_use=None, **model_kwargs):
    from sam3.model.sam3_video_predictor import Sam3VideoPredictorMultiGPU

    return Sam3VideoPredictorMultiGPU(
        *model_args, gpus_to_use=gpus_to_use, **model_kwargs
    )
