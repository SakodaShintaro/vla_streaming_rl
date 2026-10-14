# SPDX-License-Identifier: MIT
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

from ..agents.prompt import THINK_CLOSE, assistant_turn, read_action_reply, user_turn
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


class VLMActorCriticWithActionValue(NetworkInterface):
    """VLM backbone + DiffusionPolicy + Action Value critic.

    Every tick the VLM reads the tick's frame and text on top of the episode's
    conversation so far and writes the reply the zero-shot controller would --
    its action, after thinking on every ``thinking_interval``-th tick. The
    conversation lives in the VLM's KV cache: the system prompt stays, the
    turns behind it are kept first-in first-out up to ``window_tokens``, and a
    tick only runs the tokens it adds. The state is
    read off that same run -- the activations of what the tick added, its reply
    included -- so the heads see what the VLM saw, thought and proposed. The
    VLM is frozen, so what it gives a tick -- the representation and the
    action its reply names -- is computed once, when the tick is acted on, and
    kept in the replay buffer: learning reads it back and never runs the VLM
    nor reads any text. Two
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
        parse_action_text,
        text_action_names: list[str],
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
        # 返答の行動とヘッドの行動を行動価値で選ぶときの余裕。ヘッドの行動の価値が返答の行動を
        # これより大きく上回るときだけ、ヘッドの行動を実行する
        self.select_margin = vla_config.select_margin
        self.parse_action_text = parse_action_text
        self.action_low = action_low
        self.action_high = action_high
        self.decision_fps = decision_fps
        # 思考を書かせる間隔（ステップ数）。エピソードの最初のステップから、この間隔ごと
        assert vla_config.thinking_interval >= 1, vla_config.thinking_interval
        self.thinking_interval = vla_config.thinking_interval

        # 返答を書く VLM。会話は KV キャッシュに積み、システムプロンプトを残して、その後ろを
        # 直近 window_tokens トークンまで FIFO で持つ
        device = "cuda"
        self.generator = ChainGenerator(vla_config, text_action_names, torch.device(device))
        # 行動の候補が決まっている環境では、返答をその候補のどれかに絞って書かせる
        self.constrained = len(text_action_names) > 0
        self.processor = self.generator.processor
        self.device = device
        self.vlm_hidden_size = self.generator.hidden_size

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
        # このエピソードで行動を選んだ回数と、そのうち返答の行動を選んだ回数。表示用
        self._decisions_num = 0
        self._vlm_chosen_num = 0

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

    def _text_action(self, reply: str, like: torch.Tensor) -> tuple[torch.Tensor, bool, str]:
        """返答が指す行動を、方策の正規化した行動空間のチャンク (1, horizon, action_dim) に
        したものと、読めたかどうか、行動として書かれていた文。"""
        scale = (self.action_high - self.action_low) / 2.0
        bias = (self.action_high + self.action_low) / 2.0
        answer, action, parse_ok = read_action_reply(
            reply, self.parse_action_text, self.action_low, self.action_high
        )
        chunk = torch.from_numpy((action - bias) / scale).to(like.device, like.dtype)
        return chunk.view(1, 1, -1).expand(-1, self.horizon, -1), parse_ok, answer

    def _format_action(self, net_action: torch.Tensor) -> str:
        """正規化した1ステップの行動を、環境の行動空間の数値で書いたもの。"""
        scale = (self.action_high - self.action_low) / 2.0
        bias = (self.action_high + self.action_low) / 2.0
        env_action = net_action.float().cpu().numpy() * scale + bias
        return "[" + ", ".join(f"{value:+.2f}" for value in env_action) + "]"

    @torch.inference_mode()
    def infer(self, data: InferInput) -> InferResult:
        return self._act(data.system_text, data.turn_text, data.episode_step_seq, data.warmup)

    @torch.inference_mode()
    def answer(self, request: str) -> str:
        """いまの会話の続きに ``request`` を user の発言として足し、自由な文で返答を
        書かせる。返答のうち、思考より後ろを返す。"""
        assert self._exchange is not None, "there is no conversation to answer on yet"
        self._exchange.append({"role": "user", "content": [{"type": "text", "text": request}]})
        chain = self.generator.generate(self._exchange, self.generator.thinking_enabled, False)
        self._exchange.append(assistant_turn(chain.text))
        return chain.text.rsplit(THINK_CLOSE, 1)[-1].strip()

    def compute_loss(self, data: ReplayBufferData) -> LossResult:
        # 次の状態は最後のステップに保存された表現と返答の行動から作る
        terms = self._loss_terms(data, data.states[:, -1], data.reply_actions[:, -1])
        return LossResult(loss=terms.total_loss, info=terms.info)

    def infer_and_compute_loss(
        self, data: ReplayBufferData, system_text: str, turn_text: str
    ) -> InferLossResult:
        # 最後のステップはまだ行動を決めていないので、ここで返答を書いて表現を作る
        with torch.inference_mode():
            infer_result = self._act(system_text, turn_text, data.episode_step, False)
        terms = self._loss_terms(
            data,
            infer_result.stored_state.unsqueeze(0).clone(),
            infer_result.reply_action.unsqueeze(0).clone(),
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
        system_text: str,
        turn_text: str,
        episode_step: torch.Tensor,
        warmup: bool,
    ) -> InferResult:
        """最後のステップの行動を決める。そのステップのフレームと数値のターンを会話に積んで
        返答を書かせ、その計算の活性から状態を作り、返答の行動とヘッドの行動を行動価値で選ぶ。"""
        assert episode_step.shape[0] == 1, "Batch size must be 1 for inference"
        step = float(episode_step[0, -1, 0])
        if step == 0.0 or self._exchange is None:
            # エピソードの最初のステップ。前のエピソードの会話を捨てて、システムプロンプトから
            # 積み直す
            self.generator.reset_cache()
            self._exchange = [
                {"role": "system", "content": [{"type": "text", "text": system_text}]}
            ]
            self._decisions_num = 0
            self._vlm_chosen_num = 0
        self._exchange.append(user_turn(step / self.decision_fps, self._frame, turn_text))
        # 一定の間隔のステップで、思考を書いてから返答させる。そのあいだは行動だけを書かせる
        thinking = self.generator.thinking_enabled and int(step) % self.thinking_interval == 0
        chain = self.generator.generate(self._exchange, thinking, self.constrained)
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
        vlm_action, parsed, answer = self._text_action(reply, head_action)
        q_head = self.value_head.scalar_value(state, head_action).item()
        q_vlm = self.value_head.scalar_value(state, vlm_action).item()
        # バッファを貯めている間は返答の行動だけで動く
        vlm_chosen = warmup or q_head - q_vlm <= self.select_margin
        self._decisions_num += 1
        self._vlm_chosen_num += int(vlm_chosen)
        answer = answer if parsed else f"(unparsed: {answer})"

        return InferResult(
            action=vlm_action if vlm_chosen else head_action,
            value_report={
                **self.value_head.value_report(critic_out.output),
                "select/q_vlm": q_vlm,
                "select/q_head": q_head,
                "select/vlm_chosen": float(vlm_chosen),
                "vlm/parse_failed": float(not parsed),
                "vlm/prompt_tokens": float(prompt_tokens),
                "vlm/completion_tokens": float(completion_tokens),
                "vlm/msec": msec,
            },
            rnn_state=self._dummy_state.clone(),
            features=state,
            stored_state=representation,
            text=reply,
            reply_action=vlm_action[0, 0],
            text_status=(
                f"in {prompt_tokens} tok   out {completion_tokens} tok   {msec:.0f} ms   "
                f"parse {'ok' if parsed else 'failed'}"
            ),
            exchange=exchange,
            selection_status=(
                f"{'warmup: VLM only' if warmup else 'chosen by Q'}. "
                f"VLM chosen {self._vlm_chosen_num}/{self._decisions_num} this episode."
            ),
            selection_rows=[
                ("VLM", f"{answer}  {self._format_action(vlm_action[0, 0])}", q_vlm, vlm_chosen),
                ("head", self._format_action(head_action[0, 0]), q_head, not vlm_chosen),
            ],
        )

    def _loss_terms(
        self,
        data: ReplayBufferData,
        next_representation: torch.Tensor,
        next_reply_actions: torch.Tensor,
    ) -> LossTerms:
        """リプレイバッチの損失。状態は、行動を決めたステップで保存した VLM の表現から作る。
        現在の状態はチャンクの手前のステップのもので、次の状態は ``next_representation``
        （最後のステップのもの）から作る。``next_reply_actions`` (B, action_dim) は最後の
        ステップの返答が指す行動。"""
        # 次の状態で実行される行動は、推論のときと同じく、そのステップの返答の行動とヘッドの
        # 行動から行動価値で選んだもの
        with torch.inference_mode():
            next_state = self._state(next_representation)
            next_action, _ = self.policy_head.get_action(next_state)
            vlm_action = next_reply_actions.unsqueeze(1).expand(-1, self.horizon, -1)
            q_head = self.value_head.scalar_value(next_state, next_action)
            q_vlm = self.value_head.scalar_value(next_state, vlm_action)
            vlm_chosen = q_head - q_vlm <= self.select_margin
            next_action = torch.where(vlm_chosen.view(-1, 1, 1), vlm_action, next_action)
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
        """保存した表現 (B, num_state_queries, hidden_size) を射影して、ヘッドが読む状態にする。"""
        projected = self.state_out_proj(representation.to(torch.float32))
        return projected.flatten(start_dim=1)
