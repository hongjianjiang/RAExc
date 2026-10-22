#!/usr/bin/env python3
import argparse
import itertools
import json
from pathlib import Path


def classify_word(word, window_length, distinguished_value):
    return len(word) >= window_length and word[-window_length] == distinguished_value


def enumerate_words(alphabet, min_length, max_length):
    for length in range(min_length, max_length + 1):
        for word in itertools.product(alphabet, repeat=length):
            yield list(word)


def add_labeled_word(bucket, word, window_length, distinguished_value):
    bucket[tuple(word)] = classify_word(word, window_length, distinguished_value)


def build_train_continuations(window_length, distinguished_value, other_value, profile):
    if profile == "other_only":
        for continuation_length in range(window_length):
            yield tuple([other_value] * continuation_length)
        return

    if profile != "mixed_cover":
        raise ValueError(f"Unsupported train profile: {profile}")

    yielded = set()
    yield ()
    yielded.add(())

    for continuation_length in range(1, window_length):
        candidates = [
            tuple([other_value] * continuation_length),
            tuple([distinguished_value] * continuation_length),
            tuple(
                distinguished_value if idx % 2 == 0 else other_value
                for idx in range(continuation_length)
            ),
            tuple(
                other_value if idx % 2 == 0 else distinguished_value
                for idx in range(continuation_length)
            ),
        ]

        for flip_idx in range(continuation_length):
            all_other = [other_value] * continuation_length
            all_other[flip_idx] = distinguished_value
            candidates.append(tuple(all_other))

            all_distinguished = [distinguished_value] * continuation_length
            all_distinguished[flip_idx] = other_value
            candidates.append(tuple(all_distinguished))

        for continuation in candidates:
            if continuation in yielded:
                continue
            yielded.add(continuation)
            yield continuation


def build_train_words(window_length, distinguished_value, other_value, profile):
    alphabet = [distinguished_value, other_value]
    labeled = {}

    # Cover all short prefixes explicitly so the learner cannot ignore the
    # "too short to accept" region near the start state.
    for word in enumerate_words(alphabet, 0, window_length - 1):
        add_labeled_word(labeled, word, window_length, distinguished_value)

    # For every suffix-memory state s in Sigma^k, append a deterministic
    # family of continuations. The `other_only` profile uses the original
    # all-other probes. The stronger `mixed_cover` profile adds richer mixed
    # suffixes so the learner sees each state under multiple continuation
    # shapes rather than only the degenerate all-other case.
    continuations = list(
        build_train_continuations(
            window_length=window_length,
            distinguished_value=distinguished_value,
            other_value=other_value,
            profile=profile,
        )
    )
    for state in itertools.product(alphabet, repeat=window_length):
        state_word = list(state)
        for continuation in continuations:
            word = state_word + list(continuation)
            add_labeled_word(labeled, word, window_length, distinguished_value)

    return labeled


def build_eval_words(window_length, distinguished_value, other_value, eval_max_len):
    alphabet = [distinguished_value, other_value]
    labeled = {}
    for word in enumerate_words(alphabet, 0, eval_max_len):
        add_labeled_word(labeled, word, window_length, distinguished_value)
    return labeled


def split_labeled_words(labeled):
    pos = []
    neg = []
    for word, is_positive in labeled.items():
        if is_positive:
            pos.append(list(word))
        else:
            neg.append(list(word))
    pos.sort()
    neg.sort()
    return {"pos": pos, "neg": neg}


def recommended_state_sweep(window_length):
    claim = 2 ** window_length
    candidates = [
        max(4, 2 ** max(0, window_length - 2)),
        max(8, 2 ** max(0, window_length - 1)),
        claim,
        min(160, claim + claim // 2),
    ]
    return sorted(set(candidates))


def payload_from_args(args):
    train_split = split_labeled_words(
        build_train_words(
            window_length=args.window_length,
            distinguished_value=args.distinguished_value,
            other_value=args.other_value,
            profile=args.train_profile,
        )
    )
    eval_split = split_labeled_words(
        build_eval_words(
            window_length=args.window_length,
            distinguished_value=args.distinguished_value,
            other_value=args.other_value,
            eval_max_len=args.eval_max_len,
        )
    )

    claim = 2 ** args.window_length
    return {
        "task": "nth_from_end_distinguished",
        "description": (
            "Accept exactly those words whose symbol window_length positions from the end "
            "is the distinguished value. This is the classic suffix-memory family whose "
            "minimal DFA stores the recent window as a de Bruijn-style state."
        ),
        "window_length": args.window_length,
        "max_len": args.eval_max_len,
        "distinguished_value": args.distinguished_value,
        "other_value": args.other_value,
        "alphabet": [args.distinguished_value, args.other_value],
        "registers": 0,
        "constants": [args.distinguished_value],
        "theory": "Integer",
        "edge_limit_type": "at_most",
        "edge_limit_num": 2,
        "train_profile": args.train_profile,
        "train_construction": (
            "all short words plus all suffix states with all-other continuations"
            if args.train_profile == "other_only"
            else (
                "all short words plus all suffix states with a mixed continuation cover: "
                "all-other, all-distinguished, alternating, and single-flip suffix probes"
            )
        ),
        "eval_construction": "exhaustive over the binary alphabet up to eval_max_len",
        "min_active_states_claim": claim,
        "min_active_states_reason": (
            "For window_length=k, every binary suffix in Sigma^k induces a distinct right-language, "
            "so the intended recognizer needs 2^k active suffix-memory states."
        ),
        "train": train_split,
        "eval": eval_split,
        "pos": train_split["pos"],
        "neg": train_split["neg"],
        "recommended_state_sweep": recommended_state_sweep(args.window_length),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate a suffix-memory benchmark dataset for local-search scalability experiments."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window_length", type=int, required=True)
    parser.add_argument("--eval_max_len", type=int, default=None)
    parser.add_argument("--distinguished_value", type=int, default=1)
    parser.add_argument("--other_value", type=int, default=2)
    parser.add_argument(
        "--train_profile",
        choices=["other_only", "mixed_cover"],
        default="other_only",
    )
    args = parser.parse_args()

    if args.window_length < 1:
        raise ValueError("--window_length must be positive")
    if args.distinguished_value == args.other_value:
        raise ValueError("distinguished and other values must differ")
    if args.distinguished_value == 0 or args.other_value == 0:
        raise ValueError("This benchmark avoids 0 entirely")
    if args.eval_max_len is None:
        args.eval_max_len = 2 * args.window_length - 1
    if args.eval_max_len < args.window_length:
        raise ValueError("--eval_max_len must be at least --window_length")

    payload = payload_from_args(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "task": payload["task"],
                "window_length": payload["window_length"],
                "eval_max_len": payload["max_len"],
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
