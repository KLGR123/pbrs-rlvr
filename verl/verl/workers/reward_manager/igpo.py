"""
IGPO Reward Manager.

Converts pre-computed IG rewards (stored in non_tensor_batch during multi-turn
rollout) and a task-specific outcome reward into a token-level reward tensor
that the IGPO advantage estimator can consume.

Reward placement strategy
-------------------------
Given a response string consisting of T turns separated by an assistant
separator token, the manager places:

  - ig_rewards[t]  at the last response token of turn t  (for t = 0..T-2)
  - outcome_reward at the last valid response token       (final turn T-1)

The outcome reward is computed by calling self.compute_score, which the user
supplies (or falls back to igpo_compute_score for built-in task types).

Sentinel for zero IG
--------------------
When an IG value is exactly 0.0 the manager stores a small sentinel
(IG_ZERO_SENTINEL = 1e-10) instead. This ensures the advantage estimator can
still locate the turn boundary via the non-zero reward check, without adding
meaningful reward signal.

Task-type routing
-----------------
The compute_score callable receives:
    (data_source, solution_str, ground_truth, extra_info)
and should return either a float or a dict with at least {"score": float}.
The caller can pass any compatible function; igpo_tasks.igpo_compute_score
is the default that routes by task type.

Usage
-----
    from verl.workers.reward_manager.igpo import IGPORewardManager

    reward_mgr = IGPORewardManager(
        tokenizer=tokenizer,
        num_examine=1,
        compute_score=my_score_fn,   # optional, defaults to igpo_compute_score
        reward_fn_key="data_source",
        turn_separator="\\n<|im_start|>assistant\\n",
        outcome_reward_scale=1.0,
    )
    reward_dict = reward_mgr(data_batch, return_dict=True)
"""

import re
import string
from collections import defaultdict
from typing import Any, Optional

import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

# Small non-zero sentinel so the advantage estimator can locate turn boundaries
# even when the raw IG reward is exactly 0.
IG_ZERO_SENTINEL: float = 1e-10


def _find_turn_end_token_indices(
    response_str: str,
    token_ids: list[int],
    tokenizer,
    turn_separator: str,
) -> list[int]:
    """Return the last token index of each assistant turn in response_str.

    Uses the tokenizer's offset_mapping for character-to-token alignment,
    avoiding errors caused by subword tokenization boundaries.

    Args:
        response_str: Full decoded response string (all turns concatenated).
        token_ids: Token IDs of the response (after the prompt).
        tokenizer: HuggingFace tokenizer.
        turn_separator: String that separates turns in the conversation.
            Typically "\\n<|im_start|>assistant\\n".

    Returns:
        List of 0-based token indices, one per turn, pointing to the last
        token of that turn's content.
    """
    encoding = tokenizer(
        response_str,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offset_mapping: list[tuple[int, int]] = encoding["offset_mapping"]
    tokens_size = len(token_ids)

    # Find character-level end positions of each turn
    sep_positions: list[int] = []
    search_pos = 0
    while True:
        pos = response_str.find(turn_separator, search_pos)
        if pos == -1:
            break
        sep_positions.append(pos)
        search_pos = pos + 1

    if not sep_positions:
        # Single turn: last token is the end of the only turn
        return [tokens_size - 1]

    turn_char_ends: list[int] = []
    # Content before first separator
    if sep_positions[0] > 0:
        turn_char_ends.append(sep_positions[0])
    # Content between consecutive separators
    for k, sep_pos in enumerate(sep_positions):
        turn_start = sep_pos + len(turn_separator)
        if k + 1 < len(sep_positions):
            turn_char_ends.append(sep_positions[k + 1])
        else:
            turn_char_ends.append(len(response_str))

    # Map each turn's last character position to a token index
    def _char_pos_to_token_idx(char_pos: int) -> int:
        for i, (cs, ce) in enumerate(offset_mapping):
            if cs <= char_pos < ce:
                return i
            if char_pos < cs:
                return max(0, i - 1)
        return tokens_size - 1

    result: list[int] = []
    for char_end in turn_char_ends:
        if char_end > 0:
            idx = _char_pos_to_token_idx(char_end - 1)
        else:
            idx = 0
        result.append(min(idx, tokens_size - 1))
    return result


@register("igpo")
class IGPORewardManager(AbstractRewardManager):
    """Token-level reward manager for IGPO multi-turn training.

    Reads IG rewards stored in non_tensor_batch["ig_rewards"] and outcome
    rewards from self.compute_score, then places them at appropriate token
    positions in the reward tensor.

    Args:
        tokenizer: HuggingFace tokenizer used to decode response tokens and
            compute offset mappings for turn-boundary detection.
        num_examine: Number of batches to print to stdout for debugging.
        compute_score: Callable(data_source, solution_str, ground_truth,
            extra_info) → float | dict. Defaults to igpo_compute_score.
        reward_fn_key: Key in non_tensor_batch that carries the data source
            identifier used for routing in compute_score.
        turn_separator: Separator string that marks the start of each
            assistant turn in the decoded response. Default matches the
            Qwen-style chat template.
        outcome_reward_scale: Scalar multiplier applied to the outcome
            reward before placing it in the tensor (default 1.0).
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int = 0,
        compute_score=None,
        reward_fn_key: str = "data_source",
        turn_separator: str = "\n<|im_start|>assistant\n",
        outcome_reward_scale: float = 1.0,
        config = None,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.turn_separator = turn_separator
        self.outcome_reward_scale = outcome_reward_scale

        if compute_score is None:
            from verl.utils.reward_score.igpo_tasks import igpo_compute_score
            self.compute_score = igpo_compute_score
        else:
            self.compute_score = compute_score

    # ------------------------------------------------------------------
    # AbstractRewardManager interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        data: DataProto,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        """Compute and return the token-level reward tensor.

        Args:
            data: DataProto from the rollout. Must have:
                batch["responses"] – response token IDs (bsz, resp_len)
                batch["attention_mask"] – full sequence attention mask
                batch["prompts"] – prompt token IDs
                non_tensor_batch["ig_rewards"] – list[list[float]]
                non_tensor_batch["reward_model"]["ground_truth"] – str
                non_tensor_batch[reward_fn_key] – str (data source)
            return_dict: If True, return {"reward_tensor": ...,
                "reward_extra_info": ...}.

        Returns:
            Token-level reward tensor (bsz, resp_len) or dict.
        """
        # Check for pre-computed rm_scores (legacy path)
        rm_reward = self._extract_reward_from_rm_scores(data, return_dict)
        if rm_reward is not None:
            return rm_reward

        bsz = data.batch["responses"].shape[0]
        reward_tensor = torch.zeros_like(
            data.batch["responses"], dtype=torch.float32
        )
        reward_extra_info: dict[str, list] = defaultdict(list)

        already_printed: dict[str, int] = {}

        for i in range(bsz):
            item = data[i]

            # --- Decode response ---
            prompt_ids = item.batch["prompts"]
            prompt_len = prompt_ids.shape[-1]
            attn_mask = item.batch["attention_mask"]
            valid_prompt_len = int(attn_mask[:prompt_len].sum().item())
            valid_prompt_ids = prompt_ids[-valid_prompt_len:]

            resp_ids = item.batch["responses"]
            valid_resp_len = int(attn_mask[prompt_len:].sum().item())
            valid_resp_ids = resp_ids[:valid_resp_len]

            prompt_str = self.tokenizer.decode(
                valid_prompt_ids, skip_special_tokens=True
            )
            response_str = self.tokenizer.decode(
                valid_resp_ids, skip_special_tokens=True
            )

            ground_truth = item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = item.non_tensor_batch[self.reward_fn_key]
            extra_info = dict(item.non_tensor_batch.get("extra_info") or {})
            num_turns = item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            # IG rewards for this sample (T-1 values for T turns)
            ig_rewards_i: list[float] = (
                item.non_tensor_batch.get("ig_rewards") or []
            )

            # --- Compute outcome reward ---
            score_result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            if isinstance(score_result, dict):
                outcome_reward = float(score_result.get("score", 0.0))
                for k, v in score_result.items():
                    reward_extra_info[k].append(v)
            else:
                outcome_reward = float(score_result)
            outcome_reward *= self.outcome_reward_scale

            # --- Place rewards in token tensor ---
            token_ids_list = valid_resp_ids.tolist()
            scores = self._build_token_rewards(
                response_str=response_str,
                token_ids=token_ids_list,
                ig_rewards=ig_rewards_i,
                outcome_reward=outcome_reward,
                valid_resp_len=valid_resp_len,
            )

            for tok_pos, val in enumerate(scores):
                if val != 0.0:
                    reward_tensor[i, tok_pos] = val

            # --- Logging ---
            data_source_key = str(data_source)
            if data_source_key not in already_printed:
                already_printed[data_source_key] = 0
            if already_printed[data_source_key] < self.num_examine:
                already_printed[data_source_key] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[ig_rewards]", ig_rewards_i)
                print("[outcome_reward]", outcome_reward)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_token_rewards(
        self,
        response_str: str,
        token_ids: list[int],
        ig_rewards: list[float],
        outcome_reward: float,
        valid_resp_len: int,
    ) -> list[float]:
        """Build per-token reward list for one sample.

        Places IG rewards at the last token of each intermediate turn and
        the outcome reward at the last valid response token.

        Args:
            response_str: Decoded response string.
            token_ids: Token IDs of the response (valid portion only).
            ig_rewards: List of IG floats, length = number of intermediate turns.
            outcome_reward: Scalar outcome reward for the final turn.
            valid_resp_len: Number of valid response tokens.

        Returns:
            List of floats, length = valid_resp_len, mostly zeros.
        """
        scores = [0.0] * valid_resp_len
        if valid_resp_len == 0:
            return scores

        # Outcome reward always goes to the last valid token
        last_idx = valid_resp_len - 1
        scores[last_idx] = outcome_reward

        if not ig_rewards:
            return scores

        # Find turn-end token positions
        turn_end_indices = _find_turn_end_token_indices(
            response_str=response_str,
            token_ids=token_ids,
            tokenizer=self.tokenizer,
            turn_separator=self.turn_separator,
        )

        # Assign IG reward to each intermediate turn boundary
        # We have len(ig_rewards) intermediate turns (turns 0..T-2)
        # and turn_end_indices covers all T turns.
        num_intermediate = len(ig_rewards)
        for t, ig_val in enumerate(ig_rewards):
            if t >= len(turn_end_indices) - 1:
                # More IG rewards than intermediate turns found: skip excess
                break
            tok_pos = turn_end_indices[t]
            if tok_pos >= valid_resp_len:
                tok_pos = valid_resp_len - 1
            # Use sentinel for exact-zero IG to preserve turn-boundary info
            if ig_val == 0.0:
                ig_val = IG_ZERO_SENTINEL
            scores[tok_pos] = ig_val

        return scores
