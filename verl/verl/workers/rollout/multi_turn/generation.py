"""
Generic multi-turn rollout manager for IGPO.

Ported and generalised from igpo/scrl/llm_agent/generation.py, removing
all tool-server-specific imports and replacing them with a pluggable
ToolExecutor protocol so the same manager works for:

  - Retrieval / web-search agents
  - Code-sandbox (CodeAct) agents
  - Any custom tool backend

The manager handles:
  1. Conversation template formatting and tokenisation.
  2. Per-turn model generation via actor_rollout_wg.
  3. Response parsing (final answer vs. tool call).
  4. Tool execution via the injected ToolExecutor.
  5. Information gain reward computation (sequential or batched).
  6. Assembly of a verl-compatible DataProto for the full rollout.

Usage example
-------------
    from verl.workers.rollout.multi_turn.generation import (
        MultiTurnRolloutManager, RolloutConfig,
    )
    from verl.workers.rollout.multi_turn.info_gain import IGRewardConfig

    cfg = RolloutConfig(
        max_turns=5,
        n_samples_per_prompt=8,
        ig_config=IGRewardConfig(ig_type="log_prob_diff", use_batched_mode=True),
    )
    manager = MultiTurnRolloutManager(
        tokenizer=tokenizer,
        actor_rollout_wg=actor_wg,
        tool_executor=my_tool_executor,
        config=cfg,
    )

    # gen_batch: DataProto from the dataloader (initial prompts)
    # ground_truths: [{"ground_truth": str, ...}]  (one per prompt)
    rollout_batch, ig_rewards = manager.run_rollout_loop(gen_batch, ground_truths)
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from tensordict import TensorDict

from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.multi_turn.info_gain import IGRewardComputer, IGRewardConfig


# ---------------------------------------------------------------------------
# Tool executor protocol
# ---------------------------------------------------------------------------

class ToolExecutor(ABC):
    """Abstract base class for tool execution backends.

    Subclass this to implement retrieval search, code sandbox execution,
    or any other tool backend. The manager calls execute_batch once per
    turn with the list of pending tool calls.
    """

    @abstractmethod
    def execute_batch(
        self,
        tool_calls: list[Optional[dict]],
        extra_info: Optional[list[dict]] = None,
    ) -> list[str]:
        """Execute a batch of tool calls and return observation strings.

        Args:
            tool_calls: One entry per active sample. Each entry is a dict
                {"name": str, "arguments": dict} or None for samples that
                should be skipped (already finished / error).
            extra_info: Optional per-sample metadata (question text, etc.).

        Returns:
            List of observation strings, same length as tool_calls.
            Empty string for None/skipped entries.
        """


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RolloutConfig:
    """Configuration for MultiTurnRolloutManager.

    Attributes:
        max_turns: Maximum number of dialogue turns (including final answer).
        n_samples_per_prompt: Number of independent rollouts per prompt (n in GRPO).
        ig_config: Configuration for information gain reward computation.
        system_prompt: System prompt prepended to every conversation. If None,
            no system message is added.
        thinking_prefix: String appended after the chat template's generation
            prompt to encourage thinking (e.g. "<think>").
        tool_response_role: Chat role name for tool observations.
            "tool" for function-calling APIs, "user" for simpler setups.
        code_mode: If True, parse <code>...</code> tags instead of
            <tool_call>...</tool_call>. Set True for CodeAct-style agents.
        max_seq_len: Hard cap on the input_ids length fed to the model.
            Sequences longer than this are truncated on the left.
    """
    max_turns: int = 5
    n_samples_per_prompt: int = 1
    ig_config: IGRewardConfig = field(default_factory=IGRewardConfig)
    system_prompt: Optional[str] = None
    thinking_prefix: str = "<think>"
    tool_response_role: str = "tool"   # "tool" | "user"
    code_mode: bool = False            # True → CodeAct (<code> tags)
    max_seq_len: int = 8192


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_response(
    decoded: str,
    think: bool = True,
    code_mode: bool = False,
) -> tuple[bool, str, str]:
    """Parse a single decoded model response.

    Returns:
        (is_terminal, think_content, answer_or_tool_call)
        where answer_or_tool_call is the raw answer string when is_terminal
        is True, or a JSON-parsed tool call dict (or code str) otherwise.
    """
    if think:
        decoded = "<think>" + decoded

    has_think = "<think>" in decoded and "</think>" in decoded
    has_answer = "<answer>" in decoded and "</answer>" in decoded

    if has_think and has_answer:
        think_part = decoded.split("<think>")[1].split("</think>")[0]
        answer_part = decoded.split("<answer>")[1].split("</answer>")[0]
        return True, think_part, answer_part

    if code_mode:
        has_code = "<code>" in decoded and "</code>" in decoded
        if has_think and has_code:
            think_part = decoded.split("<think>")[1].split("</think>")[0]
            code_part = decoded.split("<code>")[1].split("</code>")[0]
            tool_call = {"name": "code_act", "arguments": {"code": code_part}}
            return False, think_part, tool_call
    else:
        has_tool = "<tool_call>" in decoded and "</tool_call>" in decoded
        if has_think and has_tool:
            think_part = decoded.split("<think>")[1].split("</think>")[0]
            raw_tc = decoded.split("<tool_call>")[1].split("</tool_call>")[0]
            try:
                tc = json.loads(raw_tc)
                if "name" in tc and "arguments" in tc:
                    return False, think_part, tc
            except (json.JSONDecodeError, KeyError):
                pass

    # Anything else: treat as terminal (truncated / malformed)
    return True, "", ""


def _build_context(
    tokenizer,
    messages: list[dict],
    thinking_prefix: str,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply chat template and tokenise into (input_ids, attn_mask, pos_ids).

    Returns 1D tensors truncated to max_seq_len on the left (keep recent
    context).
    """
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    text = text + thinking_prefix

    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"].squeeze(0)         # (L,)
    mask = enc["attention_mask"].squeeze(0)   # (L,)

    if ids.shape[0] > max_seq_len:
        ids = ids[-max_seq_len:]
        mask = mask[-max_seq_len:]

    # Position IDs: count from 0 over valid (non-pad) tokens
    pos = mask.long().cumsum(0) - 1
    pos = pos.clamp(min=0)

    return ids, mask, pos


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class MultiTurnRolloutManager:
    """Generic multi-turn rollout manager for IGPO.

    Drives the full rollout loop: generate → parse → tool call → repeat,
    computing information gain rewards along the way.

    Args:
        tokenizer: HuggingFace tokenizer.
        actor_rollout_wg: verl Ray worker group with generate_sequences and
            compute_log_prob methods.
        tool_executor: Backend for tool execution (retrieval / code sandbox).
        config: RolloutConfig instance.
    """

    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        tool_executor: ToolExecutor,
        config: Optional[RolloutConfig] = None,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.tool_executor = tool_executor
        self.config = config or RolloutConfig()
        self.ig_computer = IGRewardComputer(tokenizer, self.config.ig_config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_rollout_loop(
        self,
        gen_batch: DataProto,
        ground_truths: list[dict],
    ) -> tuple[DataProto, list[list[float]]]:
        """Run the full multi-turn rollout loop.

        Args:
            gen_batch: DataProto from the dataloader containing initial
                prompt token IDs (batch["input_ids"], etc.) and
                non_tensor_batch metadata (uid, ground_truth, data_source…).
            ground_truths: List of dicts, one per prompt, each containing
                at least {"ground_truth": str}.  The manager expands this
                by n_samples_per_prompt so every sample has its own copy.

        Returns:
            rollout_batch: DataProto with the assembled full rollout.
                batch fields: prompts, responses, input_ids, attention_mask,
                              position_ids, response_mask.
                non_tensor_batch: inherits gen_batch fields, plus
                              "ig_rewards" (list[list[float]]).
            ig_rewards: list[list[float]] — ig_rewards[i] contains the IG
                reward for each intermediate turn of sample i.
        """
        cfg = self.config
        n = cfg.n_samples_per_prompt

        # Expand ground_truths to match total samples (n per prompt)
        expanded_gts: list[dict] = []
        for gt in ground_truths:
            for _ in range(n):
                expanded_gts.append(gt)

        num_samples = len(expanded_gts)

        # Prepare GT tokens for IG computation
        gt_token_ids_list: list[list[int]] = []
        gt_token_ranges: list[tuple[int, int]] = []
        for gt_entry in expanded_gts:
            gt_text = gt_entry.get("ground_truth", "")
            # Handle multi-label answers: use first label
            if "<|answer_split|>" in gt_text:
                gt_text = gt_text.split("<|answer_split|>")[0]
            gt_text = gt_text.strip()
            token_ids, tok_range = self.ig_computer.prepare_gt_tokens(gt_text)
            gt_token_ids_list.append(token_ids)
            gt_token_ranges.append(tok_range)

        # Build initial message lists
        prompt_ids = gen_batch.batch["input_ids"]  # (bsz, prompt_len)
        query_strings = self._decode_prompts(prompt_ids, n)

        messages_list: list[list[dict]] = []
        for query in query_strings:
            msgs: list[dict] = []
            if cfg.system_prompt:
                msgs.append({"role": "system", "content": cfg.system_prompt})
            msgs.append({"role": "user", "content": query})
            messages_list.append(msgs)

        # ---------------------------------------------------------------
        # Rollout state
        # ---------------------------------------------------------------
        active: list[int] = list(range(num_samples))
        # final_texts[i] = decoded full multi-turn string when sample done
        final_texts: list[str] = [""] * num_samples

        # IG state (sequential mode)
        prev_gt_logprob: dict[int, Optional[float]] = {}  # sample_idx → float
        ig_rewards_seq: list[list[float]] = [[] for _ in range(num_samples)]

        # Batched mode: store context snapshot per turn per sample
        turn_snapshots: list[list[Optional[dict]]] = []  # [turn][sample]

        # Collect full rollout token sequences to assemble final DataProto
        # response_ids[i] = list of all response token IDs for sample i
        response_ids_list: list[list[int]] = [[] for _ in range(num_samples)]

        batched_mode = cfg.ig_config.use_batched_mode

        # ---------------------------------------------------------------
        # Turn loop
        # ---------------------------------------------------------------
        for turn_idx in range(cfg.max_turns):
            if not active:
                break

            # Build tokenised context batch for active samples
            ctx_tensors = self._build_active_context_batch(
                messages_list, active, cfg
            )  # dict of stacked tensors keyed by active index position

            # Save context snapshot for batched IG mode
            if batched_mode:
                snap: list[Optional[dict]] = [None] * num_samples
                for pos, sample_i in enumerate(active):
                    snap[sample_i] = {
                        "input_ids": ctx_tensors["input_ids"][pos].clone(),
                        "attention_mask": ctx_tensors["attention_mask"][pos].clone(),
                        "position_ids": ctx_tensors["position_ids"][pos].clone(),
                    }
                turn_snapshots.append(snap)

            # Sequential IG: compute GT log prob before generation
            if not batched_mode:
                for pos, sample_i in enumerate(active):
                    lp = self.ig_computer.compute_turn_gt_logprob(
                        context_input_ids=ctx_tensors["input_ids"][pos],
                        context_attention_mask=ctx_tensors["attention_mask"][pos],
                        context_position_ids=ctx_tensors["position_ids"][pos],
                        gt_token_ids=gt_token_ids_list[sample_i],
                        gt_token_range=gt_token_ranges[sample_i],
                        actor_rollout_wg=self.actor_rollout_wg,
                    )
                    if turn_idx == 0:
                        prev_gt_logprob[sample_i] = lp
                    else:
                        ig = self.ig_computer.compute_ig(
                            prev_gt_logprob.get(sample_i), lp
                        )
                        ig_rewards_seq[sample_i].append(
                            ig if ig is not None else 0.0
                        )
                        prev_gt_logprob[sample_i] = lp

            # Generate model response for active batch
            active_data = DataProto.from_dict({
                k: v for k, v in ctx_tensors.items()
            })
            gen_out = self._generate_with_gpu_padding(active_data)
            gen_responses = gen_out.batch["responses"]  # (|active|, resp_len)

            # Parse responses and update conversation
            still_active: list[int] = []
            tool_call_batch: list[tuple[int, dict]] = []  # (sample_idx, call)

            for pos, sample_i in enumerate(active):
                resp_ids = gen_responses[pos]
                # Decode: strip pad tokens
                valid_len = (resp_ids != self.tokenizer.pad_token_id).sum().item()
                resp_ids_valid = resp_ids[:valid_len]

                # Store response tokens
                response_ids_list[sample_i].extend(resp_ids_valid.tolist())

                decoded = self.tokenizer.decode(
                    resp_ids_valid, skip_special_tokens=False
                ).replace("<|endoftext|>", "")

                is_terminal, think_part, answer_or_call = _parse_response(
                    decoded, think=True, code_mode=cfg.code_mode
                )

                if is_terminal:
                    # Sample is done; record final text
                    ctx_text = self.tokenizer.apply_chat_template(
                        messages_list[sample_i],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    final_texts[sample_i] = ctx_text + cfg.thinking_prefix + decoded
                else:
                    # Tool call: keep sample active
                    still_active.append(sample_i)
                    # Append assistant turn to messages
                    if cfg.code_mode:
                        call = answer_or_call  # dict with "code" argument
                        code = call["arguments"]["code"]
                        messages_list[sample_i].append({
                            "role": "assistant",
                            "content": (
                                f"<think>{think_part}</think>\n"
                                f"<code>{code}</code>"
                            ),
                        })
                    else:
                        call = answer_or_call  # dict with name + arguments
                        messages_list[sample_i].append({
                            "role": "assistant",
                            "content": f"<think>{think_part}</think>",
                            "tool_calls": [{
                                "type": "function",
                                "function": call,
                            }],
                        })
                    tool_call_batch.append((sample_i, call))

            # Execute tool calls for still-active samples
            if tool_call_batch:
                calls_only = [tc for _, tc in tool_call_batch]
                observations = self.tool_executor.execute_batch(calls_only)

                for (sample_i, call), obs in zip(tool_call_batch, observations):
                    if cfg.code_mode:
                        messages_list[sample_i].append({
                            "role": "user",
                            "content": f"<code_response>{obs}</code_response>",
                        })
                    else:
                        messages_list[sample_i].append({
                            "role": cfg.tool_response_role,
                            "name": call.get("name", ""),
                            "content": obs,
                        })

            active = still_active

        # Any sample still active after max_turns: record as terminal
        for sample_i in active:
            ctx_text = self.tokenizer.apply_chat_template(
                messages_list[sample_i],
                add_generation_prompt=True,
                tokenize=False,
            )
            final_texts[sample_i] = ctx_text

        # ---------------------------------------------------------------
        # Batched IG computation (if enabled)
        # ---------------------------------------------------------------
        if batched_mode and turn_snapshots:
            ig_rewards_final = self.ig_computer.compute_all_ig_batched(
                turn_context_snapshots=turn_snapshots,
                gt_token_ids_list=gt_token_ids_list,
                gt_token_ranges=gt_token_ranges,
                actor_rollout_wg=self.actor_rollout_wg,
            )
        else:
            ig_rewards_final = ig_rewards_seq

        # ---------------------------------------------------------------
        # Assemble output DataProto
        # ---------------------------------------------------------------
        rollout_batch = self._assemble_rollout_batch(
            gen_batch=gen_batch,
            response_ids_list=response_ids_list,
            ig_rewards=ig_rewards_final,
            expanded_gts=expanded_gts,
            n=n,
        )

        return rollout_batch, ig_rewards_final

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _decode_prompts(
        self, prompt_ids: torch.Tensor, n: int
    ) -> list[str]:
        """Decode prompts and expand by n_samples_per_prompt."""
        decoded = []
        for ids in prompt_ids:
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            # Extract user content from chat-formatted prompt
            if "<|im_start|>user\n" in text:
                content = text.split("<|im_start|>user\n")[1].split("<|im_end|>")[0]
            else:
                content = text
            for _ in range(n):
                decoded.append(content)
        return decoded

    def _build_active_context_batch(
        self,
        messages_list: list[list[dict]],
        active: list[int],
        cfg: RolloutConfig,
    ) -> dict[str, torch.Tensor]:
        """Tokenise all active samples and stack into batch tensors.

        Returns dict with keys: input_ids, attention_mask, position_ids.
        All tensors shape (|active|, max_len), left-padded.
        """
        all_ids: list[torch.Tensor] = []
        all_masks: list[torch.Tensor] = []
        all_pos: list[torch.Tensor] = []

        for sample_i in active:
            ids, mask, pos = _build_context(
                self.tokenizer,
                messages_list[sample_i],
                cfg.thinking_prefix,
                cfg.max_seq_len,
            )
            all_ids.append(ids)
            all_masks.append(mask)
            all_pos.append(pos)

        max_len = max(t.shape[0] for t in all_ids)
        pad_id = self.tokenizer.pad_token_id

        batch_ids = torch.full((len(active), max_len), pad_id, dtype=torch.long)
        batch_masks = torch.zeros(len(active), max_len, dtype=torch.long)
        batch_pos = torch.zeros(len(active), max_len, dtype=torch.long)

        for k, (ids, mask, pos) in enumerate(zip(all_ids, all_masks, all_pos)):
            L = ids.shape[0]
            # Left-pad: valid tokens at the right end
            batch_ids[k, max_len - L:] = ids
            batch_masks[k, max_len - L:] = mask
            batch_pos[k, max_len - L:] = pos

        device = next(
            iter(self.actor_rollout_wg.workers_dict.values())
        ).device if hasattr(self.actor_rollout_wg, "workers_dict") else "cpu"

        # Keep on CPU; verl worker group handles device placement internally
        return {
            "input_ids": batch_ids,
            "attention_mask": batch_masks,
            "position_ids": batch_pos,
        }

    def _generate_with_gpu_padding(self, active_data: DataProto) -> DataProto:
        """Wrap generate_sequences to handle multi-GPU batch-size requirements."""
        return self.actor_rollout_wg.generate_sequences(active_data)

    def _assemble_rollout_batch(
        self,
        gen_batch: DataProto,
        response_ids_list: list[list[int]],
        ig_rewards: list[list[float]],
        expanded_gts: list[dict],
        n: int,
    ) -> DataProto:
        """Assemble the full rollout into a verl DataProto.

        Mirrors the structure produced by the standard single-turn rollout:
          batch: prompts, responses, input_ids, attention_mask, position_ids
          non_tensor_batch: uid, ground_truth (via reward_model key),
                            data_source, ig_rewards, plus any other keys
                            from gen_batch.
        """
        tokenizer = self.tokenizer
        pad_id = tokenizer.pad_token_id
        eos_id = tokenizer.eos_token_id

        # Expand gen_batch non_tensor fields by n
        src_ntb = gen_batch.non_tensor_batch  # original (one per prompt)
        bsz = len(response_ids_list)

        # Pad response token IDs to uniform length
        max_resp = max((len(r) for r in response_ids_list), default=1)
        responses = torch.full((bsz, max_resp), pad_id, dtype=torch.long)
        for i, r in enumerate(response_ids_list):
            if r:
                rlen = len(r)
                responses[i, :rlen] = torch.tensor(r, dtype=torch.long)

        # Original prompts (already in gen_batch, expand by n)
        orig_prompts = gen_batch.batch["input_ids"]  # (num_prompts, prompt_len)
        prompt_len = orig_prompts.shape[1]
        # Expand: interleave n copies per prompt
        expanded_prompts = orig_prompts.repeat_interleave(n, dim=0)  # (bsz, prompt_len)

        # Build full input_ids = [prompt | response]
        input_ids = torch.cat([expanded_prompts, responses], dim=1)

        # Attention mask
        prompt_mask = gen_batch.batch.get(
            "attention_mask", torch.ones_like(orig_prompts)
        ).repeat_interleave(n, dim=0)  # (bsz, prompt_len)

        resp_mask = get_response_mask(
            response_id=responses,
            eos_token=eos_id,
            dtype=prompt_mask.dtype,
        )
        attention_mask = torch.cat([prompt_mask, resp_mask], dim=1)

        # Position IDs
        prompt_pos = gen_batch.batch.get(
            "position_ids",
            torch.arange(prompt_len).unsqueeze(0).expand(
                orig_prompts.shape[0], -1
            ),
        ).repeat_interleave(n, dim=0)

        last_prompt_pos = prompt_pos[:, -1:]  # (bsz, 1)
        resp_pos = torch.arange(1, max_resp + 1, dtype=prompt_pos.dtype)
        resp_pos = (last_prompt_pos + resp_pos.unsqueeze(0)) * resp_mask
        position_ids = torch.cat([prompt_pos, resp_pos], dim=1)

        # Expand non-tensor batch fields
        new_ntb: dict = {}
        for key, val in src_ntb.items():
            if isinstance(val, list):
                expanded = []
                for item in val:
                    for _ in range(n):
                        expanded.append(item)
                new_ntb[key] = expanded
            else:
                new_ntb[key] = val

        # Override / add IGPO-specific fields
        new_ntb["ig_rewards"] = ig_rewards
        new_ntb["__num_turns__"] = [
            len(ig) + 1 for ig in ig_rewards
        ]

        # Ensure reward_model/ground_truth is populated
        if "reward_model" not in new_ntb:
            new_ntb["reward_model"] = [
                {"ground_truth": gt.get("ground_truth", "")}
                for gt in expanded_gts
            ]

        batch_td = TensorDict(
            {
                "prompts": expanded_prompts,
                "responses": responses,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=bsz,
        )

        return DataProto(batch=batch_td, non_tensor_batch=new_ntb)
