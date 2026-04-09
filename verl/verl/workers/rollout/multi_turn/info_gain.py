"""
Information Gain (IG) reward computation for IGPO.

Ported and generalized from igpo/scrl/llm_agent/generation.py and
igpo/scrl/llm_agent/vectorized_gt_logprob.py.

Core math (from the IGPO paper, Section 3.2):
    $r_{i,t}^{IG} = log π_θ(a | q, o_{i,≤t}) - log π_θ(a | q, o_{i,≤t-1})$

where a is the ground-truth answer, q is the question, o_{i,≤t} is the
observation history up to turn t. In practice we compute the mean log prob
over answer tokens under teacher forcing, then take consecutive differences.

Two computation modes:
  1. Sequential  – one compute_log_prob call per turn (T calls total, done
                   inline during rollout). Simple, minimal memory.
  2. Batched     – after rollout completes, batch all T × N (turns × samples)
                   contexts into a single compute_log_prob call, enabling
                   full GPU parallelism (the "parallel IG computation"
                   described in the paper).

Usage (sequential):
    computer = IGRewardComputer(tokenizer, config)
    for each turn t:
        logprob_t = computer.compute_turn_gt_logprob(ctx, gt_ids, gt_range, wg)
        ig_t = computer.compute_ig(prev_logprob, logprob_t)
        prev_logprob = logprob_t

Usage (batched – called once after rollout):
    computer = IGRewardComputer(tokenizer, config)
    ig_rewards = computer.compute_all_ig_batched(snapshots, gt_ids_list,
                                                  gt_ranges, actor_wg)
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from verl import DataProto


@dataclass
class IGRewardConfig:
    """Configuration for information gain reward computation.

    Attributes:
        ig_type: Basis for computing IG.
            "log_prob_diff" (default, matches paper Eq.1):
                IG_t = mean_logprob_t - mean_logprob_{t-1}
            "prob_diff":
                IG_t = exp(mean_logprob_t) - exp(mean_logprob_{t-1})
        gt_prefix: Text prepended before the ground-truth answer for
            teacher-forcing context. Should match the model's answer format.
        gt_suffix: Text appended after the ground-truth answer.
        use_batched_mode: If True, delay all GT log-prob computations and
            batch them into one call after rollout ends (more GPU-efficient).
            If False, compute per turn inline (simpler, lower memory).
        ig_source: Which model to use for GT log-prob computation.
            "policy" (default): uses the current policy (actor_rollout_wg).
            "ref": uses the frozen reference model (ref_policy_wg).
    """
    ig_type: str = "log_prob_diff"
    gt_prefix: str = "\nNow there's enough information to answer\n</think>\n<answer>\n"
    gt_suffix: str = "\n</answer><|im_end|>"
    use_batched_mode: bool = False
    ig_source: str = "policy"


class IGRewardComputer:
    """Computes per-turn information gain rewards for IGPO multi-turn rollout.

    The IG reward measures the marginal increase in the model's posterior
    probability assigned to the ground-truth answer after seeing one more
    turn of tool interaction.

    Args:
        tokenizer: HuggingFace tokenizer (must support offset_mapping).
        config: IGRewardConfig instance. Uses defaults if None.
    """

    def __init__(self, tokenizer, config: Optional[IGRewardConfig] = None):
        self.tokenizer = tokenizer
        self.config = config or IGRewardConfig()
        self._pad_id = tokenizer.pad_token_id

    # ------------------------------------------------------------------
    # Ground-truth tokenization
    # ------------------------------------------------------------------

    def prepare_gt_tokens(
        self, ground_truth: str
    ) -> tuple[list[int], tuple[int, int]]:
        """Tokenize the GT answer with teacher-forcing wrapper.

        Returns the full token list (prefix + answer + suffix) and the
        half-open token index range [start, end) of the actual answer
        tokens within that list. Only answer tokens are used for the
        mean log-prob computation.

        Args:
            ground_truth: Raw answer string.

        Returns:
            (token_ids, (gt_tok_start, gt_tok_end))
        """
        prefix = self.config.gt_prefix
        suffix = self.config.gt_suffix
        full_text = f"{prefix}{ground_truth}{suffix}"

        encoding = self.tokenizer(
            full_text,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        token_ids: list[int] = encoding["input_ids"].squeeze(0).tolist()
        offset_mapping: list[tuple[int, int]] = (
            encoding["offset_mapping"].squeeze(0).tolist()
        )

        gt_char_start = len(prefix)
        gt_char_end = len(prefix) + len(ground_truth)

        gt_tok_start: Optional[int] = None
        gt_tok_end: Optional[int] = None
        for idx, (char_s, char_e) in enumerate(offset_mapping):
            if gt_tok_start is None and char_e > gt_char_start:
                gt_tok_start = idx
            if char_s < gt_char_end and char_e > 0:
                gt_tok_end = idx + 1

        if gt_tok_start is None:
            gt_tok_start = len(token_ids)
        if gt_tok_end is None:
            gt_tok_end = len(token_ids)

        return token_ids, (gt_tok_start, gt_tok_end)

    # ------------------------------------------------------------------
    # Sequential mode – compute one turn at a time
    # ------------------------------------------------------------------

    def compute_turn_gt_logprob(
        self,
        context_input_ids: torch.Tensor,      # (ctx_len,) 1D
        context_attention_mask: torch.Tensor,  # (ctx_len,) 1D
        context_position_ids: torch.Tensor,    # (ctx_len,) 1D
        gt_token_ids: list[int],
        gt_token_range: tuple[int, int],
        actor_rollout_wg,
    ) -> Optional[float]:
        """Compute mean log prob of GT answer given the current context.

        Appends gt_token_ids to the context, calls compute_log_prob on the
        actor worker group, and returns the mean log prob over the answer
        token slice defined by gt_token_range.

        Args:
            context_input_ids: Token IDs of the current conversation context.
            context_attention_mask: Attention mask for the context.
            context_position_ids: Position IDs for the context.
            gt_token_ids: Full GT sequence (prefix + answer + suffix) as ints.
            gt_token_range: (start, end) index range of answer tokens within
                gt_token_ids.
            actor_rollout_wg: verl Ray worker group exposing compute_log_prob.

        Returns:
            Mean log prob (float) or None if gt range is empty / result is nan.
        """
        start, end = gt_token_range
        if start >= end:
            return None

        device = context_input_ids.device
        gt_tensor = torch.tensor(gt_token_ids, dtype=torch.long, device=device)
        gt_len = len(gt_token_ids)

        # Assemble full sequence: [context | gt]
        input_ids = torch.cat([context_input_ids, gt_tensor]).unsqueeze(0)  # (1, L)
        attention_mask = torch.cat([
            context_attention_mask,
            torch.ones(gt_len, dtype=context_attention_mask.dtype, device=device),
        ]).unsqueeze(0)

        last_pos = int(context_position_ids.max().item())
        gt_pos = torch.arange(
            last_pos + 1, last_pos + 1 + gt_len,
            dtype=context_position_ids.dtype,
            device=device,
        )
        position_ids = torch.cat([context_position_ids, gt_pos]).unsqueeze(0)

        data = DataProto.from_dict({
            "prompts": context_input_ids.unsqueeze(0),
            "responses": gt_tensor.unsqueeze(0),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        })

        result = actor_rollout_wg.compute_log_prob(data)
        # old_log_probs shape: (1, gt_len)
        log_probs = result.batch["old_log_probs"][0, start:end]
        mean_lp = float(log_probs.mean().item())

        if math.isnan(mean_lp) or math.isinf(mean_lp):
            return None
        return mean_lp

    # ------------------------------------------------------------------
    # IG computation from consecutive log probs
    # ------------------------------------------------------------------

    def compute_ig(
        self,
        prev_logprob: Optional[float],
        curr_logprob: Optional[float],
    ) -> Optional[float]:
        """Compute IG from two consecutive GT log probs.

        Args:
            prev_logprob: Mean log prob before the current turn (turn t-1).
            curr_logprob: Mean log prob after the current turn (turn t).

        Returns:
            IG reward (float) or None if either input is None / result is nan.
        """
        if prev_logprob is None or curr_logprob is None:
            return None

        if self.config.ig_type == "log_prob_diff":
            ig = curr_logprob - prev_logprob
        else:  # "prob_diff"
            ig = math.exp(curr_logprob) - math.exp(prev_logprob)

        if math.isnan(ig) or math.isinf(ig):
            return None
        return ig

    # ------------------------------------------------------------------
    # Batched mode – compute all turns in one call after rollout
    # ------------------------------------------------------------------

    def compute_all_ig_batched(
        self,
        turn_context_snapshots: list[list[Optional[dict]]],
        gt_token_ids_list: list[list[int]],
        gt_token_ranges: list[tuple[int, int]],
        actor_rollout_wg,
        log_prob_method: str = "compute_log_prob",
        log_prob_key: str = "old_log_probs",
    ) -> list[list[float]]:
        """Batch-compute IG rewards for all turns and samples in one call.

        Collects all valid (turn, sample) context pairs across the entire
        rollout, concatenates GT tokens, and issues a single batched
        compute_log_prob call. This achieves the "parallel IG computation"
        described in the IGPO paper.

        Args:
            turn_context_snapshots: turn_context_snapshots[t][i] is a dict
                {"input_ids": Tensor(ctx_len,),
                 "attention_mask": Tensor(ctx_len,),
                 "position_ids": Tensor(ctx_len,)}
                for sample i at turn t, or None if sample i was finished.
            gt_token_ids_list: gt_token_ids_list[i] = full GT token list for
                sample i.
            gt_token_ranges: gt_token_ranges[i] = (start, end) answer token
                range for sample i.
            actor_rollout_wg: verl Ray worker group to call for log-prob
                computation (can be the policy or ref worker group).
            log_prob_method: Method name on actor_rollout_wg to call.
                "compute_log_prob" for policy; "compute_ref_log_prob" for ref.
            log_prob_key: Key to read from the returned DataProto.
                "old_log_probs" for policy; "ref_log_prob" for ref.

        Returns:
            ig_rewards[i] = list of IG floats for sample i, one per
            intermediate turn (T-1 values for T total turns). Missing turns
            produce 0.0.
        """
        num_turns = len(turn_context_snapshots)
        num_samples = len(gt_token_ids_list)

        # --- Collect valid (turn, sample) pairs ---
        valid_pairs: list[tuple[int, int]] = []
        raw_contexts: list[dict] = []
        raw_gt_ids: list[list[int]] = []

        for t, contexts in enumerate(turn_context_snapshots):
            for i, ctx in enumerate(contexts):
                if ctx is None:
                    continue
                gt_ids = gt_token_ids_list[i]
                gt_range = gt_token_ranges[i]
                if gt_range[0] >= gt_range[1]:
                    continue
                valid_pairs.append((t, i))
                raw_contexts.append(ctx)
                raw_gt_ids.append(gt_ids)

        if not valid_pairs:
            return [[] for _ in range(num_samples)]

        bsz = len(valid_pairs)
        device = raw_contexts[0]["input_ids"].device

        # --- Build batched DataProto (pad all sequences to max length) ---
        # Each item: context tokens followed by GT tokens, right-padded.
        ctx_lens = [len(c["input_ids"]) for c in raw_contexts]
        gt_lens = [len(g) for g in raw_gt_ids]
        full_lens = [c + g for c, g in zip(ctx_lens, gt_lens)]
        max_full = max(full_lens)
        max_ctx = max(ctx_lens)
        max_gt = max(gt_lens)

        batched_input_ids = torch.full((bsz, max_full), self._pad_id,
                                       dtype=torch.long, device=device)
        batched_attn_mask = torch.zeros(bsz, max_full, dtype=torch.long, device=device)
        batched_pos_ids = torch.zeros(bsz, max_full, dtype=torch.long, device=device)
        batched_prompts = torch.full((bsz, max_ctx), self._pad_id,
                                     dtype=torch.long, device=device)
        batched_responses = torch.full((bsz, max_gt), self._pad_id,
                                       dtype=torch.long, device=device)

        for k, (ctx, gt_ids) in enumerate(zip(raw_contexts, raw_gt_ids)):
            c_ids = ctx["input_ids"]   # (ctx_len,)
            c_mask = ctx["attention_mask"]
            c_pos = ctx["position_ids"]
            gt_tensor = torch.tensor(gt_ids, dtype=torch.long, device=device)
            cl = len(c_ids)
            gl = len(gt_ids)

            batched_input_ids[k, :cl] = c_ids
            batched_input_ids[k, cl:cl + gl] = gt_tensor
            batched_attn_mask[k, :cl] = c_mask
            batched_attn_mask[k, cl:cl + gl] = 1

            last_pos = int(c_pos.max().item())
            gt_pos = torch.arange(last_pos + 1, last_pos + 1 + gl,
                                  dtype=c_pos.dtype, device=device)
            batched_pos_ids[k, :cl] = c_pos
            batched_pos_ids[k, cl:cl + gl] = gt_pos

            # Left-pad prompts so the valid tokens are right-aligned
            batched_prompts[k, max_ctx - cl:] = c_ids
            batched_responses[k, :gl] = gt_tensor

        batch_data = DataProto.from_dict({
            "prompts": batched_prompts,
            "responses": batched_responses,
            "input_ids": batched_input_ids,
            "attention_mask": batched_attn_mask,
            "position_ids": batched_pos_ids,
        })

        result = getattr(actor_rollout_wg, log_prob_method)(batch_data)
        log_probs = result.batch[log_prob_key]  # (bsz, max_gt)

        # --- Extract per-(turn, sample) mean log probs ---
        pair_logprob: dict[tuple[int, int], Optional[float]] = {}
        for k, (turn_idx, sample_idx) in enumerate(valid_pairs):
            start, end = gt_token_ranges[sample_idx]
            lp = float(log_probs[k, start:end].mean().item())
            if math.isnan(lp) or math.isinf(lp):
                pair_logprob[(turn_idx, sample_idx)] = None
            else:
                pair_logprob[(turn_idx, sample_idx)] = lp

        # --- Compute IG as consecutive log-prob differences ---
        ig_rewards: list[list[float]] = [[] for _ in range(num_samples)]
        for i in range(num_samples):
            prev_lp: Optional[float] = None
            for t in range(num_turns):
                curr_lp = pair_logprob.get((t, i))
                if t == 0:
                    # First turn: initialise reference prob, no IG yet
                    prev_lp = curr_lp
                    continue
                if curr_lp is not None and prev_lp is not None:
                    ig = self.compute_ig(prev_lp, curr_lp)
                    ig_rewards[i].append(ig if ig is not None else 0.0)
                    prev_lp = curr_lp
                # If sample was already done at turn t, append nothing
                # (shorter ig list means fewer turns completed).

        return ig_rewards
