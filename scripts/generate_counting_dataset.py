#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def dedupe_sequences(sequences):
    seen = set()
    unique = []
    for seq in sequences:
        key = tuple(seq)
        if key in seen:
            continue
        seen.add(key)
        unique.append(seq)
    return unique


def tail_variants(length, values):
    if length <= 0:
        return [[]]

    values = list(values)
    candidates = []

    for value in values[: min(3, len(values))]:
        candidates.append([value] * length)

    if len(values) >= 2:
        candidates.append([values[i % 2] for i in range(length)])
        candidates.append([values[(i + 1) % 2] for i in range(length)])

    if len(values) >= 3:
        candidates.append([values[i % len(values)] for i in range(length)])
        candidates.append([values[(2 * i + 1) % len(values)] for i in range(length)])

    if len(values) >= 4:
        candidates.append([values[(length + i) % len(values)] for i in range(length)])

    if len(values) >= 5:
        candidates.append([values[(i * i + length) % len(values)] for i in range(length)])

    return dedupe_sequences(candidates)


def select_eval_tail_lengths(budget):
    if budget <= 0:
        return []
    return sorted({1, min(2, budget), min(5, budget), budget})


def round_robin_take(families, limit):
    ordered_names = list(families.keys())
    iterators = {name: iter(families[name]) for name in ordered_names}
    exhausted = {name: False for name in ordered_names}
    result = []
    seen = set()

    while limit is None or len(result) < limit:
        added = False
        for name in ordered_names:
            if exhausted[name]:
                continue

            while True:
                try:
                    seq = next(iterators[name])
                except StopIteration:
                    exhausted[name] = True
                    break

                key = tuple(seq)
                if key in seen:
                    continue
                seen.add(key)
                result.append(seq)
                added = True
                break

            if limit is not None and len(result) >= limit:
                return result

        if not added:
            break

    return result


def round_robin_take_weighted(families, limit):
    ordered_names = []
    iterators = {}
    exhausted = {}
    for name, (iterable, weight) in families.items():
        iterators[name] = iter(iterable)
        exhausted[name] = False
        ordered_names.extend([name] * max(1, int(weight)))

    result = []
    seen = set()

    while limit is None or len(result) < limit:
        added = False
        for name in ordered_names:
            if exhausted[name]:
                continue

            while True:
                try:
                    seq = next(iterators[name])
                except StopIteration:
                    exhausted[name] = True
                    break

                key = tuple(seq)
                if key in seen:
                    continue
                seen.add(key)
                result.append(seq)
                added = True
                break

            if limit is not None and len(result) >= limit:
                return result

        if not added:
            break

    return result


def encode_tail(length, values, code):
    base = len(values)
    digits = []
    for _ in range(length):
        digits.append(values[code % base])
        code //= base
    return digits


def positive_tail_generator(tail_length, values):
    total_codes = len(values) ** tail_length
    for code in range(total_codes):
        yield encode_tail(tail_length, values, code)


def build_positive_samples(prefix_length, max_len, values, target_count, distinguished_value):
    tail_max = max_len - prefix_length
    families = {
        f"tail_len_{tail_length}": (
            ([distinguished_value] * prefix_length) + tail
            for tail in positive_tail_generator(tail_length, values)
        )
        for tail_length in range(1, tail_max + 1)
    }
    return round_robin_take(families, target_count)


def wrong_prefix_single_other_family(prefix_length, max_len, values, distinguished_value):
    for witness_value in values:
        for distinguished_count in range(max_len):
            if distinguished_count == prefix_length:
                continue
            yield ([distinguished_value] * distinguished_count) + [witness_value]


def all_distinguished_family(max_len, distinguished_value):
    for distinguished_count in range(max_len + 1):
        yield [distinguished_value] * distinguished_count


def wrong_prefix_tail_family(prefix_length, max_len, values, distinguished_value):
    tail_max = max_len - prefix_length
    for distinguished_count in range(max_len):
        if distinguished_count == prefix_length:
            continue
        budget = max_len - distinguished_count
        if budget <= 0:
            continue
        for tail_length in sorted({1, min(2, budget), min(3, budget), budget}):
            total_codes = len(values) ** tail_length
            for code in range(total_codes):
                yield ([distinguished_value] * distinguished_count) + encode_tail(
                    tail_length, values, code
                )


def distinguished_after_suffix_family(prefix_length, max_len, values, distinguished_value):
    tail_max = max_len - prefix_length
    for left_length in range(1, tail_max):
        right_budget = tail_max - left_length - 1
        total_codes = len(values) ** left_length
        for left_code in range(total_codes):
            left_tail = encode_tail(left_length, values, left_code)
            yield ([distinguished_value] * prefix_length) + left_tail + [distinguished_value]
            if right_budget <= 0:
                continue
            right_length = min(2, right_budget)
            right_total_codes = len(values) ** right_length
            for right_code in range(right_total_codes):
                right_tail = encode_tail(right_length, values, right_code)
                yield ([distinguished_value] * prefix_length) + left_tail + [
                    distinguished_value
                ] + right_tail


def premature_other_then_distinguished_family(prefix_length, max_len, values, distinguished_value):
    anchor_prefixes = sorted({0, prefix_length // 4, prefix_length // 2, prefix_length - 1})
    for distinguished_count in anchor_prefixes:
        if distinguished_count < 0 or distinguished_count >= prefix_length:
            continue
        budget = max_len - distinguished_count - 1
        if budget <= 0:
            continue
        for tail_length in sorted({1, min(3, budget)}):
            total_codes = len(values) ** tail_length
            for code in range(total_codes):
                tail = encode_tail(tail_length, values, code)
                remaining = max_len - distinguished_count - tail_length
                distinguished_after = min(3, remaining)
                yield ([distinguished_value] * distinguished_count) + tail + (
                    [distinguished_value] * distinguished_after
                )


def wrong_prefix_valid_suffix_family(distinguished_count, tail_lengths, values, distinguished_value):
    for tail_length in tail_lengths:
        for tail in positive_tail_generator(tail_length, values):
            yield ([distinguished_value] * distinguished_count) + tail


def hard_wrong_prefix_valid_suffix_families(prefix_length, max_len, values, distinguished_value):
    families = {}
    for distinguished_count in range(max_len):
        if distinguished_count == prefix_length:
            continue
        budget = max_len - distinguished_count
        if budget <= 0:
            continue
        tail_lengths = sorted({1, min(2, budget), min(5, budget), budget})
        families[
            f"wrong_prefix_valid_suffix_{distinguished_count}"
        ] = wrong_prefix_valid_suffix_family(
            distinguished_count,
            tail_lengths,
            values,
            distinguished_value,
        )
    return families


def valid_prefix_bad_suffix_family(prefix_length, max_len, values, distinguished_value):
    yield [distinguished_value] * prefix_length
    for left_length in range(1, max_len - prefix_length):
        left_total_codes = len(values) ** left_length
        for left_code in range(left_total_codes):
            left_tail = encode_tail(left_length, values, left_code)
            word = ([distinguished_value] * prefix_length) + left_tail + [distinguished_value]
            if len(word) <= max_len:
                yield word


def fixed_block_positive_samples(
    prefix_length,
    total_length,
    block_length,
    values,
    target_count,
    distinguished_value,
):
    suffix_zeros = total_length - prefix_length - block_length
    if suffix_zeros < 0:
        raise ValueError("total_length must be at least prefix_length + block_length")

    return round_robin_take(
        {
            "fixed_block_pos": (
                ([distinguished_value] * prefix_length)
                + block
                + ([distinguished_value] * suffix_zeros)
                for block in positive_tail_generator(block_length, values)
            )
        },
        target_count,
    )


def shifted_block_family(start, total_length, block_length, values, distinguished_value):
    suffix_zeros = total_length - start - block_length
    for block in positive_tail_generator(block_length, values):
        yield ([distinguished_value] * start) + block + ([distinguished_value] * suffix_zeros)


def shifted_block_families(prefix_length, total_length, block_length, values, distinguished_value):
    families = {}
    max_start = total_length - block_length
    for start in range(max_start + 1):
        if start == prefix_length:
            continue
        families[f"shifted_block_{start}"] = shifted_block_family(
            start,
            total_length,
            block_length,
            values,
            distinguished_value,
        )
    return families


def broken_block_families(prefix_length, total_length, block_length, values, distinguished_value):
    suffix_zeros = total_length - prefix_length - block_length
    if suffix_zeros < 0:
        raise ValueError("total_length must be at least prefix_length + block_length")

    families = {}
    for distinguished_index in range(block_length):
        families[f"broken_block_distinguished_{distinguished_index}"] = (
            ([distinguished_value] * prefix_length)
            + block[:distinguished_index]
            + [distinguished_value]
            + block[distinguished_index + 1 :]
            + ([distinguished_value] * suffix_zeros)
            for block in positive_tail_generator(block_length, values)
        )
    return families


def extra_other_families(prefix_length, total_length, block_length, values, distinguished_value):
    suffix_zeros = total_length - prefix_length - block_length
    if suffix_zeros < 1:
        return {}

    families = {}
    for extra_offset in range(suffix_zeros):
        families[f"extra_other_suffix_{extra_offset}"] = (
            ([distinguished_value] * prefix_length)
            + block
            + ([distinguished_value] * extra_offset)
            + [extra_value]
            + ([distinguished_value] * (suffix_zeros - extra_offset - 1))
            for block in positive_tail_generator(block_length, values[: min(3, len(values))])
            for extra_value in values[: min(3, len(values))]
        )
    return families


def shorter_exact_block_families(prefix_length, total_length, block_length, values, distinguished_value):
    suffix_distinguished = total_length - prefix_length - block_length
    if suffix_distinguished < 0:
        raise ValueError("total_length must be at least prefix_length + block_length")

    families = {}
    for missing_suffix in range(1, suffix_distinguished + 1):
        kept_suffix = suffix_distinguished - missing_suffix
        families[f"shorter_exact_suffix_missing_{missing_suffix}"] = (
            ([distinguished_value] * prefix_length)
            + block
            + ([distinguished_value] * kept_suffix)
            for block in positive_tail_generator(block_length, values)
        )

    for visible_block_length in range(1, block_length):
        families[f"truncated_block_{visible_block_length}"] = (
            ([distinguished_value] * prefix_length) + block_prefix
            for block_prefix in positive_tail_generator(visible_block_length, values)
        )

    families["shorter_all_distinguished"] = (
        [distinguished_value] * length for length in range(prefix_length, total_length)
    )
    return families


def resolve_split_targets(total_target, positive_fraction):
    positive_target = max(1, math.floor(total_target * positive_fraction))
    negative_target = max(1, total_target - positive_target)
    return positive_target, negative_target


def build_train_split(
    prefix_length,
    max_len,
    values,
    negative_multiplier,
    distinguished_value,
    total_target=None,
    positive_fraction=0.5,
    profile="simple",
):
    tail_max = max_len - prefix_length
    if tail_max < 1:
        raise ValueError("max_len must be at least prefix_length + 1")

    if total_target is not None:
        pos_target, neg_target = resolve_split_targets(total_target, positive_fraction)
        pos = build_positive_samples(prefix_length, max_len, values, pos_target, distinguished_value)
    else:
        pos = dedupe_sequences(
            ([distinguished_value] * prefix_length) + tail
            for tail_length in range(1, tail_max + 1)
            for tail in tail_variants(tail_length, values)[:6]
        )
        neg_target = max(4, negative_multiplier * len(pos))

    if profile == "hard_frontier":
        neg_families = hard_wrong_prefix_valid_suffix_families(
            prefix_length, max_len, values, distinguished_value
        )
        neg_families["valid_prefix_bad_suffix"] = valid_prefix_bad_suffix_family(
            prefix_length, max_len, values, distinguished_value
        )
        neg_families["all_distinguished"] = all_distinguished_family(
            max_len, distinguished_value
        )
    else:
        neg_families = {
            "wrong_prefix_single_other": wrong_prefix_single_other_family(
                prefix_length, max_len, values, distinguished_value
            ),
            "all_distinguished": all_distinguished_family(max_len, distinguished_value),
            "wrong_prefix_rich_tail": wrong_prefix_tail_family(
                prefix_length, max_len, values, distinguished_value
            ),
            "distinguished_after_suffix": distinguished_after_suffix_family(
                prefix_length, max_len, values, distinguished_value
            ),
        }
    return {
        "pos": pos,
        "neg": round_robin_take(neg_families, neg_target),
    }


def build_eval_split(
    prefix_length,
    max_len,
    values,
    negative_multiplier,
    distinguished_value,
    total_target=None,
    positive_fraction=0.5,
    profile="simple",
):
    tail_max = max_len - prefix_length
    if tail_max < 1:
        raise ValueError("max_len must be at least prefix_length + 1")

    if total_target is not None:
        pos_target, neg_target = resolve_split_targets(total_target, positive_fraction)
        pos = build_positive_samples(prefix_length, max_len, values, pos_target, distinguished_value)
    else:
        pos = dedupe_sequences(
            ([distinguished_value] * prefix_length) + tail
            for tail_length in range(1, tail_max + 1)
            for tail in tail_variants(tail_length, values)
        )
        neg_target = max(4, negative_multiplier * len(pos))

    if profile == "hard_frontier":
        neg_families = hard_wrong_prefix_valid_suffix_families(
            prefix_length, max_len, values, distinguished_value
        )
        neg_families["valid_prefix_bad_suffix"] = valid_prefix_bad_suffix_family(
            prefix_length, max_len, values, distinguished_value
        )
        neg_families["all_distinguished"] = all_distinguished_family(
            max_len, distinguished_value
        )
    else:
        neg_families = {
            "all_distinguished": all_distinguished_family(max_len, distinguished_value),
            "wrong_prefix_tail": wrong_prefix_tail_family(
                prefix_length, max_len, values, distinguished_value
            ),
            "distinguished_after_suffix": distinguished_after_suffix_family(
                prefix_length, max_len, values, distinguished_value
            ),
            "premature_other_then_distinguished": premature_other_then_distinguished_family(
                prefix_length, max_len, values, distinguished_value
            ),
        }
    return {
        "pos": pos,
        "neg": round_robin_take(neg_families, neg_target),
    }


def dataset_payload(args):
    if args.task == "fixed_nonzero_block":
        return fixed_block_dataset_payload(args)

    train_split = build_train_split(
        args.prefix_length,
        args.max_len,
        args.train_nonzero_values,
        args.train_negative_multiplier,
        args.distinguished_value,
        args.train_total,
        args.train_positive_fraction,
        args.profile,
    )
    eval_split = build_eval_split(
        args.prefix_length,
        args.max_len,
        args.eval_nonzero_values,
        args.eval_negative_multiplier,
        args.distinguished_value,
        args.eval_total,
        args.eval_positive_fraction,
        args.profile,
    )
    lower_bound = args.prefix_length + 3

    return {
        "task": "distinguished_prefix_then_other_suffix",
        "profile": args.profile,
        "description": (
            "Accept exactly those sequences with exactly prefix_length leading distinguished symbols, "
            "followed by one or more other values, and no later distinguished symbol."
        ),
        "prefix_length": args.prefix_length,
        "max_len": args.max_len,
        "suffix_max_len": args.max_len - args.prefix_length,
        "distinguished_value": args.distinguished_value,
        "registers": args.registers,
        "constants": [args.distinguished_value],
        "theory": "Integer",
        "edge_limit_type": "at_most",
        "edge_limit_num": 2,
        "min_active_states_claim": lower_bound,
        "min_active_states_reason": (
            "The intended recognizer needs one pre-switch state for each distinguished-prefix count "
            "0..prefix_length, plus an accepting other-suffix state and a rejecting sink."
        ),
        "train_nonzero_values": args.train_nonzero_values,
        "eval_nonzero_values": args.eval_nonzero_values,
        "train_negative_multiplier": args.train_negative_multiplier,
        "eval_negative_multiplier": args.eval_negative_multiplier,
        "train_total": args.train_total,
        "eval_total": args.eval_total,
        "train_positive_fraction": args.train_positive_fraction,
        "eval_positive_fraction": args.eval_positive_fraction,
        "train": train_split,
        "eval": eval_split,
        "pos": train_split["pos"],
        "neg": train_split["neg"],
        "recommended_state_sweep": [
            10,
            20,
            40,
            60,
            80,
            args.prefix_length,
            lower_bound,
            lower_bound + 17,
        ],
    }


def fixed_block_dataset_payload(args):
    train_split = build_fixed_block_split(
        prefix_length=args.prefix_length,
        total_length=args.max_len,
        block_length=args.block_length,
        values=args.train_nonzero_values,
        distinguished_value=args.distinguished_value,
        profile=args.fixed_block_profile,
        total_target=args.train_total,
        positive_fraction=args.train_positive_fraction,
    )
    eval_split = build_fixed_block_split(
        prefix_length=args.prefix_length,
        total_length=args.max_len,
        block_length=args.block_length,
        values=args.eval_nonzero_values,
        distinguished_value=args.distinguished_value,
        profile=args.fixed_block_profile,
        total_target=args.eval_total,
        positive_fraction=args.eval_positive_fraction,
    )
    lower_bound = args.max_len + 2

    return {
        "task": "fixed_other_block",
        "fixed_block_profile": args.fixed_block_profile,
        "description": (
            "Accept exactly those sequences of length max_len whose first prefix_length symbols are the "
            "distinguished value, whose next block_length symbols are other values, and whose remaining "
            "suffix symbols are again the distinguished value."
        ),
        "prefix_length": args.prefix_length,
        "max_len": args.max_len,
        "block_length": args.block_length,
        "distinguished_value": args.distinguished_value,
        "registers": args.registers,
        "constants": [args.distinguished_value],
        "theory": "Integer",
        "edge_limit_type": "at_most",
        "edge_limit_num": 2,
        "min_active_states_claim": lower_bound,
        "min_active_states_reason": (
            "The intended recognizer must count to the exact start of the other-value block, track the block, "
            "and then verify the exact distinguished-value suffix on a fixed-length word."
        ),
        "train_nonzero_values": args.train_nonzero_values,
        "eval_nonzero_values": args.eval_nonzero_values,
        "train_negative_multiplier": args.train_negative_multiplier,
        "eval_negative_multiplier": args.eval_negative_multiplier,
        "train_total": args.train_total,
        "eval_total": args.eval_total,
        "train_positive_fraction": args.train_positive_fraction,
        "eval_positive_fraction": args.eval_positive_fraction,
        "train": train_split,
        "eval": eval_split,
        "pos": train_split["pos"],
        "neg": train_split["neg"],
        "recommended_state_sweep": [
            10,
            20,
            40,
            60,
            80,
            args.prefix_length,
            lower_bound,
            lower_bound + 20,
        ],
    }


def build_fixed_block_split(
    prefix_length,
    total_length,
    block_length,
    values,
    distinguished_value,
    profile,
    total_target=None,
    positive_fraction=0.5,
):
    if total_target is None:
        raise ValueError("fixed_nonzero_block requires explicit --train_total/--eval_total")

    pos_target, neg_target = resolve_split_targets(total_target, positive_fraction)
    pos = fixed_block_positive_samples(
        prefix_length=prefix_length,
        total_length=total_length,
        block_length=block_length,
        values=values,
        target_count=pos_target,
        distinguished_value=distinguished_value,
    )

    shifted_families = shifted_block_families(
        prefix_length=prefix_length,
        total_length=total_length,
        block_length=block_length,
        values=values,
        distinguished_value=distinguished_value,
    )

    broken_families = broken_block_families(
        prefix_length=prefix_length,
        total_length=total_length,
        block_length=block_length,
        values=values,
        distinguished_value=distinguished_value,
    )
    extra_families = extra_other_families(
        prefix_length=prefix_length,
        total_length=total_length,
        block_length=block_length,
        values=values,
        distinguished_value=distinguished_value,
    )

    if profile == "harder":
        shorter_families = shorter_exact_block_families(
            prefix_length=prefix_length,
            total_length=total_length,
            block_length=block_length,
            values=values,
            distinguished_value=distinguished_value,
        )
        weighted_families = {}
        for name, family in shifted_families.items():
            weighted_families[name] = (family, 1)
        for name, family in broken_families.items():
            weighted_families[name] = (family, 8)
        for name, family in extra_families.items():
            weighted_families[name] = (family, 6)
        for name, family in shorter_families.items():
            weighted_families[name] = (family, 6)
        weighted_families["all_distinguished_fixed_length"] = (
            ([distinguished_value] * total_length for _ in range(1)),
            3,
        )
        neg = round_robin_take_weighted(weighted_families, neg_target)
    else:
        neg_families = {}
        neg_families.update(shifted_families)
        neg_families.update(broken_families)
        neg_families.update(extra_families)
        neg_families["all_distinguished_fixed_length"] = (
            [distinguished_value] * total_length for _ in range(1)
        )
        neg = round_robin_take(neg_families, neg_target)

    return {
        "pos": pos,
        "neg": neg,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate a frozen counting dataset for local-search state-scalability experiments."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task",
        choices=["zero_prefix_then_nonzero_suffix", "fixed_nonzero_block"],
        default="zero_prefix_then_nonzero_suffix",
    )
    parser.add_argument("--prefix_length", type=int, default=100)
    parser.add_argument("--max_len", type=int, default=110)
    parser.add_argument("--block_length", type=int, default=4)
    parser.add_argument("--distinguished_value", type=int, default=1)
    parser.add_argument(
        "--fixed_block_profile",
        choices=["baseline", "harder"],
        default="harder",
        help="Negative-family mix for fixed-block tasks.",
    )
    parser.add_argument("--registers", type=int, default=0)
    parser.add_argument(
        "--profile",
        choices=["simple", "hard_frontier"],
        default="simple",
        help="Sampling profile for the same target language. hard_frontier concentrates on hard wrong-prefix confusers.",
    )
    parser.add_argument(
        "--train_nonzero_values",
        type=int,
        nargs="+",
        default=[2, -2, 3, -3, 4],
    )
    parser.add_argument(
        "--eval_nonzero_values",
        type=int,
        nargs="+",
        default=[5, -4, 6, -5, 7],
    )
    parser.add_argument("--train_negative_multiplier", type=int, default=4)
    parser.add_argument("--eval_negative_multiplier", type=int, default=4)
    parser.add_argument("--train_total", type=int, default=None)
    parser.add_argument("--eval_total", type=int, default=None)
    parser.add_argument("--train_positive_fraction", type=float, default=0.5)
    parser.add_argument("--eval_positive_fraction", type=float, default=0.5)
    args = parser.parse_args()

    if args.prefix_length < 1:
        raise ValueError("--prefix_length must be positive")
    if args.max_len <= args.prefix_length:
        raise ValueError("--max_len must be greater than --prefix_length")
    if args.block_length < 1:
        raise ValueError("--block_length must be positive")
    if args.task == "fixed_nonzero_block" and args.max_len < args.prefix_length + args.block_length:
        raise ValueError("--max_len must be at least prefix_length + block_length for fixed_nonzero_block")
    if args.distinguished_value == 0 or 0 in args.train_nonzero_values or 0 in args.eval_nonzero_values:
        raise ValueError("This benchmark avoids 0 entirely; choose nonzero distinguished/other values.")
    if args.distinguished_value in args.train_nonzero_values or args.distinguished_value in args.eval_nonzero_values:
        raise ValueError("Other-value sets must not contain the distinguished value.")
    if args.train_total is not None and args.train_total < 2:
        raise ValueError("--train_total must be at least 2")
    if args.eval_total is not None and args.eval_total < 2:
        raise ValueError("--eval_total must be at least 2")
    if not 0.0 < args.train_positive_fraction < 1.0:
        raise ValueError("--train_positive_fraction must be in (0, 1)")
    if not 0.0 < args.eval_positive_fraction < 1.0:
        raise ValueError("--eval_positive_fraction must be in (0, 1)")

    payload = dataset_payload(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "task": payload["task"],
                "prefix_length": payload["prefix_length"],
                "max_len": payload["max_len"],
                "registers": payload["registers"],
                "train_pos": len(payload["train"]["pos"]),
                "train_neg": len(payload["train"]["neg"]),
                "eval_pos": len(payload["eval"]["pos"]),
                "eval_neg": len(payload["eval"]["neg"]),
                "min_active_states_claim": payload["min_active_states_claim"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
