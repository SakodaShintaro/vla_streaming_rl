# SPDX-License-Identifier: MIT
import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from vla_streaming_rl.networks.modules.backbone import SpatialTemporalEncoder
from vla_streaming_rl.networks.modules.image_processor import ImageProcessor
from vla_streaming_rl.networks.modules.reward_processor import RewardProcessor
from vla_streaming_rl.wrappers import _car_racing_parse_action

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=Path, default="./local/image/ep_00000001")
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    images_dir = args.images_dir
    num_images = args.num_images
    batch_size = args.batch_size

    # Transform for image preprocessing
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
        ]
    )

    image_list = []

    if images_dir.is_file():
        cap = cv2.VideoCapture(str(images_dir))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image_list.append(frame)
            if len(image_list) >= num_images:
                break
        cap.release()
    else:
        image_path_list = sorted(images_dir.glob("*.png"))
        image_path_list = image_path_list[:num_images]
        for i, image_path in enumerate(image_path_list):
            image = cv2.imread(str(image_path))
            image_list.append(image)

    print(f"Loaded {len(image_list)} images from {images_dir}")

    # Combine images as sequence and move to GPU
    image_list = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in image_list]
    image_list = [transform(image) for image in image_list]
    images_sequence = torch.stack(image_list)  # shape: (sequence_length, 3, height, width)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_sequence = images_sequence.to(device)
    print(f"Images sequence shape: {images_sequence.shape}")
    print(f"Using device: {device}")

    # Initialize encoder
    print("\nInitializing Encoder...")
    observation_space_shape = (3, 96, 96)
    seq_len = images_sequence.shape[0]

    image_processor = ImageProcessor(observation_space_shape)
    image_processor = image_processor.to(device)

    hidden_image_dim = image_processor.output_shape[0]
    reward_processor = RewardProcessor(embed_dim=hidden_image_dim)
    reward_processor = reward_processor.to(device)

    encoder = SpatialTemporalEncoder(
        image_processor=image_processor,
        reward_processor=reward_processor,
        seq_len=seq_len,
        n_layer=1,
        action_dim=3,
        temporal_model_type="transformer",
        use_image_only=False,
    )
    encoder = encoder.to(device)

    print(f"\n{encoder.__class__.__name__}")

    batched_images = images_sequence.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
    seq_len = images_sequence.shape[0]

    obs_z_shape = image_processor.output_shape
    obs_z = torch.zeros(batch_size, seq_len, *obs_z_shape, device=device)
    rnn_state = encoder.init_state().to(device)
    rnn_state = rnn_state.repeat(batch_size, *([1] * (rnn_state.dim() - 1)))
    action_sequence = torch.randn((batch_size, seq_len, 3), device=device)
    reward_sequence = torch.randn((batch_size, seq_len, 1), device=device)

    time_list = []
    trial_num = 10

    for _ in range(trial_num):
        start = time.time()
        representation, rnn_state, action_text = encoder(
            batched_images, obs_z, action_sequence, reward_sequence, rnn_state
        )
        end = time.time()
        elapsed_msec = (end - start) * 1000
        time_list.append(elapsed_msec)

    time_list.sort()
    print(time_list)
    remove_num = len(time_list) // 10
    time_list = time_list[remove_num:-remove_num]
    mean_time = np.mean(time_list)
    std_time = np.std(time_list)

    print(f"  Batch size: {batch_size}")
    print(f"  Representation shape: {representation.shape}")
    print(f"  Elapsed time: {mean_time:.1f} ms (±{std_time:.1f} ms)")
    print(f"  Action text: '{action_text}'")
    action_values, parse_success = _car_racing_parse_action(action_text)
    print(
        f"  Parsed action: steering={action_values[0]:.3f}, gas={action_values[1]:.3f}, braking={action_values[2]:.3f}, success={parse_success}"
    )
