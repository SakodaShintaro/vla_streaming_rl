# SPDX-License-Identifier: MIT
import torch
import torch.nn.functional as F

from vla_streaming_rl.agents.prompt import PromptBuilder
from vla_streaming_rl.agents.subtask_cycle import SubtaskCycle

from .chain_generator import ChainGenerator


class CoTBatch:
    """The planner's reply as the policy reads it: the activations behind the
    subtask written every ``steps_per_chain`` environment steps, held in between.

    When a subtask is judged and when the next is planned is
    :class:`SubtaskCycle`'s, the loop the zero-shot controller runs too, over
    :class:`ChainGenerator`, the generator that controller's local backend
    writes with -- so what differs between the two agents is who acts on the
    subtask and nothing else. What is added here is the pooling of the planner's
    activations to one step's read.
    """

    def __init__(
        self,
        model_id: str,
        load_in_4bit: bool,
        tokens_per_step: int,
        max_len: int,
        temperature: float,
        steps_per_chain: int,
        use_cuda_graph: bool,
        prompt_budget: int,
        prompt_builder: PromptBuilder,
        device: torch.device,
    ) -> None:
        assert tokens_per_step >= 1, f"tokens_per_step must be positive; got {tokens_per_step}"
        assert steps_per_chain >= 1, f"steps_per_chain must be positive; got {steps_per_chain}"
        assert max_len >= tokens_per_step, (
            f"max_len {max_len} below {tokens_per_step}: the pool would stretch a chain "
            "shorter than one step's read"
        )
        # Thinking off: with the <think> block left open the model spends the
        # chain reasoning about the request rather than about the scene.
        self.generator = ChainGenerator(
            model_id=model_id,
            load_in_4bit=load_in_4bit,
            max_len=max_len,
            temperature=temperature,
            enable_thinking=False,
            use_cuda_graph=use_cuda_graph,
            prompt_budget=prompt_budget,
            device=device,
        )
        self.cycle = SubtaskCycle(self.generator.generate, prompt_builder, steps_per_chain)
        self.tokens_per_step = tokens_per_step
        self.device = device
        self.reset()

    def reset(self) -> None:
        """Drop the chain. The next advance plans a new subtask on the frame it
        is given; the conversation it is written into is the builder's to reset."""
        self.cycle.reset()
        self._activations = torch.zeros(
            (self.tokens_per_step, self.generator.layers_num, self.generator.hidden_size),
            dtype=torch.bfloat16,
            device=self.device,
        )

    def age(self) -> int:
        """How many environment steps ago the chain now being read was written.

        0 on the step that wrote it, up to ``steps_per_chain - 1`` on the last
        step that holds it. What the encoder needs alongside the activations:
        the same chain means something different on the frame it was written
        about than it does fifteen steps later, and nothing else in the
        observation says which of the two this is -- `episode_step` carries it
        only modulo a period the encoder cannot take.
        """
        return self.cycle.age()

    @torch.inference_mode()
    def advance(self, episode_done: bool) -> torch.Tensor:
        """This environment step's activations, the cycle judging and planning
        when due.

        Returns:
            (tokens_per_step, layers_num, hidden_size) bfloat16. The same tensor
            on every step until the next subtask is planned.
        """
        self.cycle.advance(episode_done)
        if self.cycle.planned:
            self._activations = self._read_activations(self.cycle.plan.positions)
        return self._activations

    def _read_activations(self, positions: torch.Tensor) -> torch.Tensor:
        """The whole chain, every depth kept, pooled to one step's read:
        (tokens_per_step, layers_num, hidden_size).

        ``positions`` is (chain_len, layers_num, hidden_size): each position's
        activation is the state that chose its token. A chain stops where the
        model stops it, so the positions are pooled along the chain rather than
        sliced to its tail: the whole chain reaches the policy, and a chain that
        ended early needs no padding to reach the fixed read.
        """
        pooled = F.adaptive_avg_pool1d(
            positions.to(torch.float32).permute(1, 2, 0), self.tokens_per_step
        )
        return pooled.permute(2, 0, 1).to(torch.bfloat16)

    def stats(self) -> dict:
        """What the last plan cost: the tokens it was given, the tokens it wrote,
        and the wall time the write took."""
        return {
            "input_tokens": self.cycle.plan.prompt_tokens,
            "output_tokens": len(self.cycle.plan.tokens),
            "msec": self.cycle.plan_msec,
        }

    def text(self) -> str:
        """The planner's reply as written, for logging."""
        return self.cycle.plan.text

    def exchange(self) -> list[dict]:
        """The last plan as it happened: the conversation the model was given
        and the reply it wrote back, for the render panel."""
        return self.cycle.exchange
