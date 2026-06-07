# SPDX-License-Identifier: MIT
import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim

from vla_streaming_rl.replay_buffer import ReplayBuffer, ReplayBufferData
from vla_streaming_rl.reward_processor import RewardProcessor
from vla_streaming_rl.self_forcing.goal_predictor import WorldModelGoalPredictor


class SequentialBatchSampler:
    def __init__(
        self, buffer_capacity: int, batch_size: int, k_frames: int, drop_last: bool
    ) -> None:
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.k_frames = k_frames
        self.drop_last = drop_last

    def __iter__(self):
        valid_starts = range(self.buffer_capacity - self.k_frames + 1)
        randomized_starts = torch.randperm(len(valid_starts)).tolist()

        batch = []
        for start_idx in randomized_starts:
            seq_indices = list(range(start_idx, start_idx + self.k_frames))
            batch.append(seq_indices)

            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        num_samples = self.buffer_capacity - self.k_frames + 1
        if self.drop_last:
            return num_samples // self.batch_size
        else:
            return (num_samples + self.batch_size - 1) // self.batch_size


class OnPolicyAgent:
    def __init__(
        self,
        *,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        network: nn.Module,
        network_class: str,
        on_policy_epoch: int,
        gamma: float,
        buffer_capacity: int,
        seq_len: int,
        horizon: int,
        batch_size: int,
        accumulation_steps: int,
        num_bins: int,
        max_grad_norm: float,
        use_done: bool,
        normalizing_by_return: bool,
        max_new_tokens: int,
        max_prompt_tokens: int,
        pad_token_id: int,
        buffer_device: str,
        learning_rate: float,
        goal_predictor: WorldModelGoalPredictor,
    ) -> None:
        self.on_policy_epoch = on_policy_epoch
        # action properties
        self.action_space = action_space
        self.action_dim = np.prod(action_space.shape)
        self.action_low = action_space.low
        self.action_high = action_space.high
        self.action_scale = (action_space.high - action_space.low) / 2.0
        self.action_bias = (action_space.high + action_space.low) / 2.0

        self.gamma = gamma
        self.buffer_capacity = buffer_capacity
        self.seq_len = seq_len
        self.horizon = horizon
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.device = torch.device("cuda")
        self.num_bins = num_bins
        self.max_grad_norm = max_grad_norm
        self.use_done = use_done

        # Action chunking state
        self.action_chunk = None  # (horizon, action_dim) - current action chunk
        self.chunk_step = 0  # current step within chunk

        self.reward_processor = RewardProcessor("scaling", 1.0)
        self.normalizing_by_return = normalizing_by_return

        self._uses_state_value = network_class == "actor_critic_with_state_value"
        self.network = network
        self.rnn_state = self.network.init_state().to(self.device)
        obs_z_shape = tuple(self.network.image_processor.output_shape)

        self.max_new_tokens = max_new_tokens
        self.pad_token_id = pad_token_id

        self.rb = ReplayBuffer(
            size=self.buffer_capacity,
            seq_len=self.seq_len + self.horizon,
            obs_shape=observation_space.shape,
            obs_z_shape=obs_z_shape,
            rnn_state_shape=self.rnn_state.squeeze(0).shape,
            action_shape=action_space.shape,
            output_device=self.device,
            storage_device=torch.device(buffer_device),
            max_new_tokens=self.max_new_tokens,
            max_prompt_tokens=max_prompt_tokens,
            pad_token_id=self.pad_token_id,
        )

        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.prev_logp = 0.0
        self.prev_value = 0.0
        self.prev_action_token_ids: list[int] = []
        self.prev_parse_success = True

        self.goal_predictor = goal_predictor

    @torch.inference_mode()
    def select_action(
        self,
        global_step: int,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        task_prompt: str,
    ) -> tuple[np.ndarray, dict]:
        info_dict = {}

        # Reset chunk on episode boundary
        if terminated or truncated:
            self.action_chunk = None
            self.chunk_step = 0

        # calculate train reward
        action_norm = np.linalg.norm(self.prev_action)
        if not self.normalizing_by_return:
            self.reward_processor.update(reward)
        info_dict["action_norm"] = action_norm
        info_dict["processed_reward"] = self.reward_processor.normalize(torch.tensor(reward)).item()

        # add to replay buffer
        obs_tensor = torch.from_numpy(obs).to(self.device)
        obs_z = self.network.image_processor.encode(obs_tensor.unsqueeze(0))
        obs_z = obs_z.squeeze(0)
        normalized_action = (self.prev_action - self.action_bias) / self.action_scale
        task_prompt_token_ids = self.network.tokenize_task_prompt(task_prompt)
        self.rb.add(
            obs_tensor,
            obs_z,
            reward,
            (terminated or truncated) if self.use_done else False,
            self.rnn_state.squeeze(0),
            torch.from_numpy(normalized_action).to(self.device),
            self.prev_logp,
            self.prev_value,
            self.prev_action_token_ids,
            task_prompt_token_ids,
        )

        info_dict["goal_image"] = self.goal_predictor.step(obs)
        if terminated or truncated:
            self.goal_predictor.reset()

        # Use cached action from chunk if available
        if self.action_chunk is not None and self.chunk_step < self.horizon:
            action = self.action_chunk[self.chunk_step]
            action = action * self.action_scale + self.action_bias
            action = np.clip(action, self.action_low, self.action_high)
            self.prev_action = action
            self.chunk_step += 1
            info_dict["a_logp"] = self.prev_logp
            info_dict["value"] = self.prev_value
            info_dict["chunk_step"] = self.chunk_step
            return action, info_dict

        # inference - predict new action chunk
        latest_data = self.rb.get_latest(1)
        result_dict = self.network.infer(
            latest_data.observations,
            latest_data.obs_z,
            latest_data.actions,
            latest_data.rewards,
            self.rnn_state,
            task_prompts=[task_prompt],
        )
        self.rnn_state = result_dict["rnn_state"]

        # action chunk: (B, horizon, action_dim) -> (horizon, action_dim)
        action_chunk = result_dict["action"][0].cpu().numpy()
        self.action_chunk = action_chunk
        self.chunk_step = 1

        # Use first action from chunk
        action = action_chunk[0]
        action = action * self.action_scale + self.action_bias
        action = np.clip(action, self.action_low, self.action_high)
        self.prev_action = action

        # log prob
        a_logp = result_dict["a_logp"]
        a_logp = a_logp.item()
        self.prev_logp = a_logp
        info_dict["a_logp"] = a_logp

        # value
        value = self.network.value_head.to_value(result_dict["value"]).item()
        self.prev_value = value
        info_dict["value"] = value

        # action token ids and parse success
        self.prev_action_token_ids = result_dict["action_token_ids"]
        self.prev_parse_success = result_dict["parse_success"]

        # predict next state
        info_dict["next_image"] = result_dict["next_image"]
        info_dict["next_reward"] = result_dict["next_reward"]

        info_dict["chunk_step"] = self.chunk_step
        return action, info_dict

    def step(
        self,
        global_step: int,
        obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        task_prompt: str,
    ) -> tuple[np.ndarray, dict]:
        info_dict = {}

        # train
        train_info = self._train(self.prev_value)
        info_dict.update(train_info)

        # make decision
        action, action_info = self.select_action(
            global_step, obs, reward, terminated, truncated, task_prompt
        )
        info_dict.update(action_info)

        return action, info_dict

    def on_episode_end(self, score: float, feedback_text: str) -> dict:
        return {}

    ####################
    # Internal methods #
    ####################

    def _train(self, last_value: float) -> dict:
        info_dict = {}

        if not self.rb.is_full():
            return info_dict

        buffer_data = self.rb.get_all_data()
        s = buffer_data.observations
        s_z = buffer_data.obs_z
        a = buffer_data.actions
        r = buffer_data.rewards.view(-1, 1)
        v = buffer_data.values.view(-1, 1)
        old_a_logp = buffer_data.log_probs.view(-1, 1)
        done = buffer_data.dones.view(-1, 1)
        rnn_states = buffer_data.rnn_state
        action_token_ids = buffer_data.action_token_ids
        task_prompt_token_ids_all = buffer_data.task_prompt_token_ids

        bias = torch.tensor(self.action_bias, device=self.device).view(1, -1)
        scale = torch.tensor(self.action_scale, device=self.device).view(1, -1)
        a = (a - bias) / scale

        # apply reward processing
        r = self.reward_processor.normalize(r)

        # preprocessing
        target_v = torch.zeros_like(v[:-1])
        adv = torch.zeros_like(v[:-1])
        last_advantage = 0
        for i in range(len(v) - 2, -1, -1):
            if done[i + 1]:
                last_value = 0
                last_advantage = 0
            target_v[i] = r[i + 1] + self.gamma * last_value
            delta = target_v[i] - v[i + 1]
            last_advantage = delta + self.gamma * 0.95 * last_advantage
            adv[i] = last_advantage
            last_value = v[i + 1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        ave_action_loss_list = []
        ave_value_loss_list = []
        metrics_dict = {}
        is_first_batch = True

        for _ in range(self.on_policy_epoch):
            sum_action_loss = 0.0
            sum_value_loss = 0.0
            self.optimizer.zero_grad()
            batch_idx = 0
            for indices in SequentialBatchSampler(
                self.buffer_capacity - self.horizon + 1,
                self.batch_size,
                k_frames=self.seq_len + self.horizon,
                drop_last=False,
            ):
                indices = np.array(indices, dtype=np.int64)  # [B, T]
                data = ReplayBufferData(
                    observations=s[indices],
                    obs_z=s_z[indices],
                    rewards=r[indices],
                    dones=done[indices],
                    rnn_state=rnn_states[indices],
                    actions=a[indices],
                    log_probs=old_a_logp[indices],
                    values=v[indices],
                    action_token_ids=action_token_ids[indices],
                    task_prompt_token_ids=task_prompt_token_ids_all[indices],
                )
                # Index for chunk start point (last frame of input sequence)
                chunk_start_idx = indices[:, -self.horizon - 1]
                curr_adv = adv[chunk_start_idx]
                curr_target_v = target_v[chunk_start_idx]

                if self._uses_state_value:
                    loss, activations_dict, info_dict = self.network.compute_loss(
                        data, curr_target_v, curr_adv
                    )
                else:
                    loss, activations_dict, info_dict = self.network.compute_loss(
                        data, curr_target_v.squeeze(1)
                    )

                sum_action_loss += info_dict.get("actor_loss", 0.0) * len(data.observations)
                sum_value_loss += info_dict["critic_loss"] * len(data.observations)

                scaled_loss = loss / self.accumulation_steps
                scaled_loss.backward()

                # Collect metrics on the first batch only
                if is_first_batch:
                    # Feature metrics
                    for feature_name, feature in activations_dict.items():
                        metrics_dict[f"activation_norms/{feature_name}"] = (
                            feature.norm(dim=1).mean().item()
                        )
                    is_first_batch = False

                batch_idx += 1
                if batch_idx % self.accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            ave_action_loss = sum_action_loss / self.buffer_capacity
            ave_value_loss = sum_value_loss / self.buffer_capacity
            ave_action_loss_list.append(ave_action_loss)
            ave_value_loss_list.append(ave_value_loss)

        result_dict = {
            "losses/actor_loss": np.mean(ave_action_loss_list),
            "losses/critic_loss": np.mean(ave_value_loss_list),
            **metrics_dict,
        }

        self.rb.reset()

        return result_dict
