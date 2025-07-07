# inference_main.py
# Inference-only script for trained RAVEN models

import os
import torch
import numpy as np
from analogy.nn import Model
from jactorch.parallel import JacDataParallel
from PIL import Image


def load_model(model_path, args):
    model = Model(
        model_name=args.model,
        nr_features=args.num_features,
        nr_experts=args.num_experts,
        shared_group_mlp=not args.normal_group_mlp,
        one_hot=args.one_hot,
        v2s_softmax=args.v2s_softmax,
        not_use_softmax=args.not_use_softmax,
        visual_inputs=args.use_visual_inputs,
        factor_groups=args.factor_groups,
        split_channel=args.split_channel,
        image_size=args.image_size,
        use_layer_norm=args.use_layer_norm,
        use_resnet=args.use_resnet,
        conv_hidden_dims=args.conv_hidden_dims,
        conv_repeats=args.conv_repeats,
        conv_kernels=args.conv_kernels,
        conv_residual_link=args.conv_residual_link,
        nr_visual_experts=args.num_visual_experts,
        visual_mlp_hidden_dims=args.visual_mlp_hidden_dims,
        transformed_spatial_dim=args.transformed_spatial_dim,
        mlp_transform_hidden_dims=args.mlp_transform_hidden_dims,
        exclude_angle_attr=args.exclude_angle_attr,
        symbolic_beta=args.symbolic_beta,
        prediction_beta=args.prediction_beta,
        embedding_dim=args.embedding_dim,
        embedding_hidden_dims=args.embedding_hidden_dims,
        enable_residual_block=args.enable_residual_block,
        use_ordinary_mlp=args.use_ordinary_mlp,
        enable_rb_after_experts=args.enable_rb_after_experts,
        feature_embedding_dim=args.feature_embedding_dim,
        hidden_dims=args.hidden_dims,
        reduction_groups=args.reduction_groups,
        sum_as_reduction=args.sum_as_reduction,
        lastmlp_hidden_dims=args.lastmlp_hidden_dims,
        nr_context=8,
        nr_candidates=8
    )

    if args.use_gpu and torch.cuda.is_available():
        model = JacDataParallel(model).cuda()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def preprocess_image(img):
    if isinstance(img, str):
        img = Image.open(img).convert('L')
    img = img.resize((160, 160))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr[None, None, :, :]  # shape: (1, 1, H, W)
    return torch.tensor(arr)


def run_inference(model, image_array, use_gpu=True):
    tensor = torch.tensor(image_array / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
    if use_gpu:
        tensor = tensor.cuda()
    with torch.no_grad():
        output = model(tensor)
    prediction = output.argmax(dim=1).item()
    return prediction


# Example usage:
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--input-npz', type=str, required=True, help='Path to .npz file with shape (16, 160, 160)')
    parser.add_argument('--use-gpu', action='store_true')

    # Minimal required model args for loading
    parser.add_argument('--model', default='analogy')
    parser.add_argument('--num-features', type=int, default=80)
    parser.add_argument('--num-experts', type=int, default=5)
    parser.add_argument('--use-visual-inputs', action='store_true')
    parser.add_argument('--factor-groups', type=int, default=10)
    parser.add_argument('--image-size', type=int, nargs=2, default=[160, 160])
    parser.add_argument('--conv-hidden-dims', type=int, nargs='+', default=[16, 16, 32, 32])
    parser.add_argument('--embedding-hidden-dims', type=int, nargs='+', default=[])
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[64, 32])
    parser.add_argument('--lastmlp-hidden-dims', type=int, nargs='+', default=[128])

    args = parser.parse_args()

    model = load_model(args.model_path, args)
    data = np.load(args.input_npz)['image']  # shape: (16, 160, 160)

    result = run_inference(model, data, use_gpu=args.use_gpu)
    print("Predicted answer index:", result)
