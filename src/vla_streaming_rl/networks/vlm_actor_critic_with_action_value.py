# SPDX-License-Identifier: MIT
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

from ..agents.prompt import assistant_turn, read_action_reply, user_turn
from ..replay_buffer import ReplayBufferData
from .interface import (
    EligibilityTraceInfo,
    InferInput,
    InferLossResult,
    InferResult,
    LossResult,
    NetworkInterface,
)
from .modules.chain_generator import ChainGenerator
from .modules.head_output import HeadOutput
from .modules.value_head import DistributionalValueHead

# 表示用に残す直近の会話のメッセージ数
EXCHANGE_MESSAGES = 8


@dataclass
class LossTerms:
    """リプレイバッチ1つ分の損失と、それを計算する途中で得た推論結果。"""

    total_loss: torch.Tensor
    actor_loss: torch.Tensor
    info: dict
    state: torch.Tensor
    action_chunk: torch.Tensor
    next_state: torch.Tensor
    next_action: torch.Tensor
    next_critic_out: HeadOutput


def _rows(data: ReplayBufferData) -> tuple[torch.Tensor, ...]:
    """What a replay batch holds of each tick's prompt, in the order
    ``_act`` reads them."""
    return (
        data.system_token_ids,
        data.turn_token_ids,
        data.episode_step,
    )


class VLMActorCriticWithActionValue(NetworkInterface):
    """VLM backbone + DiffusionPolicy + Action Value critic.

    Every tick the VLM reads the tick's frame and text on top of the episode's
    conversation so far and writes the reply the zero-shot controller would --
    its action. The conversation lives in the VLM's KV cache: the system prompt
    stays, the turns behind it are kept first-in first-out up to
    ``window_tokens``, and a tick only runs the tokens it adds. The state is
    read off that same run -- the activations of what the tick added, its reply
    included -- so the heads see what the VLM saw, thought and proposed. The
    VLM is frozen, so what it gives a tick is computed once, when the tick is
    acted on, and kept in the replay buffer: learning reads it back and never
    runs the VLM. Two
    candidate actions come out of a tick, the one the reply names and the policy
    head's, and the critic picks: the head's runs only where it is valued above
    the reply's by more than ``select_margin``. ``.inf`` there leaves the
    zero-shot controller alone at the wheel and ``-.inf`` the head; while the
    buffer fills it is always the reply's.

    Architecture:
    - VLM (Qwen3.5/Qwen-VL, frozen): processes images + text.
    - Stored representation: the mean across every VLM hidden state (embedding
      + each transformer layer output) at the positions the tick added,
      AdaptiveAvgPool1d over them to ``num_state_queries``.
    - State extractor: a per-token Linear projection of that representation.
    - DiffusionPolicy: denoises actions conditioned on extracted state.
    - Critic: Q(state, action) with dueling architecture.
    """

    def __init__(
        self,
        *,
        observation_space_shape: tuple[int],
        action_space_shape: tuple[int],
        value_head_factory: Callable[[int, int], DistributionalValueHead],
        horizon: int,
        critic_loss_weight: float,
        policy_head_factory,
        vla_config: DictConfig,
        pad_token_id: int,
        parse_action_text,
        action_low: np.ndarray,
        action_high: np.ndarray,
        decision_fps: float,
    ) -> None:
        super().__init__()
        # 会話は VLM の KV キャッシュが持つので、バッファから読む窓は今のステップだけでよい
        self.seq_len = 1
        # 状態の元になる表現は、行動を決めたステップでだけ計算して保存する。チャンクの途中の
        # ステップには保存されないので、毎ステップ行動を決める設定でだけ使える
        assert horizon == 1, f"the VLM network stores a state on every tick; got horizon {horizon}"
        self.horizon = horizon
        self.action_dim = action_space_shape[0]
        self.critic_loss_weight = critic_loss_weight
        # VLM は凍結して使う。状態は保存した表現から作るので、VLM には勾配が届かない
        assert not vla_config.use_lora, "the VLM is frozen in this network"
        # 返答の行動と頭の行動を行動価値で選ぶときの余裕。頭の行動の価値が返答の行動を
        # これより大きく上回るときだけ、頭の行動を実行する
        self.select_margin = vla_config.select_margin
        self.parse_action_text = parse_action_text
        self.action_low = action_low
        self.action_high = action_high
        self.decision_fps = decision_fps

        # 返答を書く VLM。会話は KV キャッシュに積み、システムプロンプトを残して、その後ろを
        # 直近 window_tokens トークンまで FIFO で持つ
        device = "cuda"
        self.generator = ChainGenerator(vla_config, torch.device(device))
        self.processor = self.generator.processor
        self.device = device
        self.vlm_hidden_size = self.generator.hidden_size
        self.pad_token_id = pad_token_id

        self.num_state_queries = vla_config.num_state_queries

        self.state_out_proj = nn.Linear(self.vlm_hidden_size, vla_config.state_out_dim).to(device)
        # AdaptiveAvgPool1d fixes the token count to num_state_queries, so
        # state_dim is determined purely by config.
        state_dim = vla_config.num_state_queries * vla_config.state_out_dim

        self.policy_head = policy_head_factory(state_dim=state_dim, action_dim=self.action_dim)

        # Critic: Q(state, action)
        self.value_head = value_head_factory(state_dim, self.action_dim)

        self._dummy_state = torch.zeros(1, 1, 1)
        # このステップのフレーム（会話に積むために覚えておく）と、表示用に残す直近の会話
        self._frame = None
        self._exchange = None

    def init_state(self) -> torch.Tensor:
        return self._dummy_state.clone()

    def stored_image_shape(self) -> tuple[int, ...]:
        """何も保存しない。フレームは会話に積んだ時点で KV キャッシュに入る。"""
        return (0,)

    def to_stored_image(self, image: torch.Tensor) -> torch.Tensor:
        """バッファには何も置かず、このステップの会話に積むフレームとして覚えておく。"""
        self._frame = image
        return torch.zeros(0)

    def stored_state_shape(self) -> tuple[int, ...]:
        """VLM の表現を ``num_state_queries`` トークンに均したもの。射影は学習するので、
        その手前を保存する。"""
        return (self.num_state_queries, self.vlm_hidden_size)

    def observe_scalar_obs(
        self,
        velocity_x: float,
        velocity_y: float,
        velocity_z: float,
        episode_return: float,
        pass_mark: float,
        remaining_return: float,
        global_step: float,
        episode_step: float,
        health: float,
    ) -> None:
        del velocity_x, velocity_y, velocity_z, episode_return, pass_mark
        del remaining_return, global_step, episode_step, health

    def tokenize(self, text: str) -> list[int]:
        return self.processor.tokenizer.encode(text, add_special_tokens=False)

    def _decode(self, token_ids: torch.Tensor) -> list[str]:
        """Strings back from their stored token IDs, (N, max_prompt_tokens)."""
        results = []
        for ids in token_ids:
            valid_ids = ids[ids != self.pad_token_id].tolist()
            results.append(self.processor.tokenizer.decode(valid_ids, skip_special_tokens=True))
        return results

    def _text_actions(self, replies: list[str], like: torch.Tensor) -> tuple[torch.Tensor, list]:
        """各返答の <action> が指す行動を、方策の正規化した行動空間のチャンク
        (B, horizon, action_dim) にしたものと、読めたかどうか。"""
        scale = (self.action_high - self.action_low) / 2.0
        bias = (self.action_high + self.action_low) / 2.0
        actions, parsed = [], []
        for reply in replies:
            _, action, parse_ok = read_action_reply(
                reply, self.parse_action_text, self.action_low, self.action_high
            )
            actions.append((action - bias) / scale)
            parsed.append(parse_ok)
        chunk = torch.from_numpy(np.stack(actions)).to(like.device, like.dtype)
        return chunk.unsqueeze(1).expand(-1, self.horizon, -1), parsed

    @torch.inference_mode()
    def infer(self, data: InferInput) -> InferResult:
        return self._act(
            data.system_token_ids_seq,
            data.turn_token_ids_seq,
            data.episode_step_seq,
            data.warmup,
        )

    def compute_loss(self, data: ReplayBufferData) -> LossResult:
        # 次の状態は最後のステップに保存された表現と返答から作る
        replies = self._decode(data.reply_token_ids[:, -1])
        terms = self._loss_terms(data, data.states[:, -1], replies)
        return LossResult(loss=terms.total_loss, info=terms.info)

    def infer_and_compute_loss(self, data: ReplayBufferData) -> InferLossResult:
        # 最後のステップはまだ行動を決めていないので、ここで返答を書いて表現を作る
        with torch.inference_mode():
            infer_result = self._act(*_rows(data), False)
        terms = self._loss_terms(
            data, infer_result.stored_state.unsqueeze(0).clone(), [infer_result.text]
        )
        # -Q(s,a) for eligibility trace backward (detached from encoder)
        neg_value_detached = -self.value_head.scalar_value(
            terms.state.detach(), terms.action_chunk.detach()
        ).mean()
        return InferLossResult(
            infer_result=infer_result,
            loss_result=LossResult(loss=terms.total_loss, info=terms.info),
            et_info=EligibilityTraceInfo(
                # Actor-only loss (no critic component)
                actor_entropy_loss=terms.actor_loss,
                neg_value=neg_value_detached,
                delta=terms.info["losses/delta"],
            ),
        )

    ####################
    # Internal methods #
    ####################

    def _act(
        self,
        system_token_ids: torch.Tensor,
        turn_token_ids: torch.Tensor,
        episode_step: torch.Tensor,
        warmup: bool,
    ) -> InferResult:
        """最後のステップの行動を決める。そのステップのフレームと数値のターンを会話に積んで
        返答を書かせ、その計算の活性から状態を作り、返答の行動と頭の行動を行動価値で選ぶ。"""
        assert system_token_ids.shape[0] == 1, "Batch size must be 1 for inference"
        step = float(episode_step[0, -1, 0])
        if step == 0.0 or self._exchange is None:
            # エピソードの最初のステップ。前のエピソードの会話を捨てて、システムプロンプトから
            # 積み直す
            self.generator.reset_cache()
            system_text = self._decode(system_token_ids[0, -1:])[0]
            self._exchange = [
                {"role": "system", "content": [{"type": "text", "text": system_text}]}
            ]
        turn_text = self._decode(turn_token_ids[0, -1:])[0]
        self._exchange.append(user_turn(step / self.decision_fps, self._frame, turn_text))
        chain = self.generator.generate(self._exchange)
        reply = chain.text
        self._exchange.append(assistant_turn(reply))
        # モデルの文脈はキャッシュが持つので、リストには差分の計算と表示に要る直近だけを残す
        self._exchange = self._exchange[-EXCHANGE_MESSAGES:]
        exchange = list(self._exchange)
        prompt_tokens, completion_tokens, msec = chain.prompt_tokens, len(chain.tokens), chain.msec
        # このステップで積んだ位置の活性を num_state_queries トークンに均す
        pooled = F.adaptive_avg_pool1d(
            chain.states.to(torch.float32).T.unsqueeze(0), self.num_state_queries
        )
        representation = pooled[0].T.to(torch.bfloat16)
        state = self._state(representation.unsqueeze(0))
        head_action, _ = self.policy_head.get_action(state)
        critic_out = self.value_head(state, head_action)
        text_action, parsed = self._text_actions([reply], head_action)
        q_head = self.value_head.scalar_value(state, head_action).item()
        q_text = self.value_head.scalar_value(state, text_action).item()
        # バッファを貯めている間は返答の行動だけで動く
        text_chosen = warmup or q_head - q_text <= self.select_margin

        return InferResult(
            action=text_action if text_chosen else head_action,
            value_report={
                **self.value_head.value_report(critic_out.output),
                "select/q_head": q_head,
                "select/q_text": q_text,
                "select/text_chosen": float(text_chosen),
                "vlm/parse_failed": float(not parsed[0]),
                "vlm/prompt_tokens": float(prompt_tokens),
                "vlm/completion_tokens": float(completion_tokens),
                "vlm/msec": msec,
            },
            rnn_state=self._dummy_state.clone(),
            features=state,
            stored_state=representation,
            text=reply,
            text_status=(
                f"in {prompt_tokens} tok   out {completion_tokens} tok   {msec:.0f} ms   "
                f"parse {'ok' if parsed[0] else 'failed'}   "
                f"{'reply' if text_chosen else 'head'} action runs"
            ),
            exchange=exchange,
        )

    def _loss_terms(
        self, data: ReplayBufferData, next_representation: torch.Tensor, next_replies: list[str]
    ) -> LossTerms:
        """リプレイバッチの損失。状態は、行動を決めたステップで保存した VLM の表現から作る。
        現在の状態はチャンクの手前のステップのもので、次の状態は ``next_representation``
        （最後のステップのもの）から作る。"""
        # 次の状態で実行される行動は、推論のときと同じく、そのステップの返答の行動と頭の
        # 行動から行動価値で選んだもの
        with torch.inference_mode():
            next_state = self._state(next_representation)
            next_action, _ = self.policy_head.get_action(next_state)
            text_action, _ = self._text_actions(next_replies, next_action)
            q_head = self.value_head.scalar_value(next_state, next_action)
            q_text = self.value_head.scalar_value(next_state, text_action)
            text_chosen = q_head - q_text <= self.select_margin
            next_action = torch.where(text_chosen.view(-1, 1, 1), text_action, next_action)
            next_critic_out = self.value_head(next_state, next_action)
        target_value = self.value_head.compute_target_value(
            next_critic_out.output,
            data.rewards[:, -self.horizon :],
            data.dones[:, -self.horizon :],
        )

        state = self._state(data.states[:, -self.horizon - 1])
        action_chunk = data.actions[:, -self.horizon :]  # (B, horizon, action_dim)
        critic_loss, critic_info = self.value_head.compute_critic_loss(
            state, action_chunk, target_value
        )
        actor_loss, actor_info = self.policy_head.compute_actor_loss(
            state, action_chunk, value_head=self.value_head
        )
        info = {f"losses/{key}": value for key, value in {**critic_info, **actor_info}.items()}
        return LossTerms(
            total_loss=self.critic_loss_weight * critic_loss + actor_loss,
            actor_loss=actor_loss,
            info=info,
            state=state,
            action_chunk=action_chunk,
            next_state=next_state,
            next_action=next_action,
            next_critic_out=next_critic_out,
        )

    def _state(self, representation: torch.Tensor) -> torch.Tensor:
        """保存した表現 (B, num_state_queries, hidden_size) を射影して、頭が読む状態にする。"""
        projected = self.state_out_proj(representation.to(torch.float32))
        return projected.flatten(start_dim=1)
