#!/usr/bin/env python3
"""Read-only real Science expert checkpoint smoke test (no training or writes)."""
import hashlib
import json
import sys
from pathlib import Path

# When invoked as `python tests/<file>.py`, Python places tests/ rather than
# the repository root at sys.path[0]. Add the project root for tools imports.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from tools.expert_inference import ExpertRunner, tokenize

CHECKPOINT = Path(sys.argv[1] if len(sys.argv) > 1 else "science.expert.pt")
EXPECTED_SOURCE_RUN = "44795411621"
EXPECTED_DOMAIN = "science"


def choose_prompt(runner, candidates, language):
    for prompt in candidates:
        tokens = tokenize(prompt)
        known = [w for w in tokens if w in runner.vocab_to_id]
        if len(known) >= 2:
            return prompt, len(known), len(tokens) - len(known)
    raise AssertionError(
        f"No {language} smoke prompt has >=2 in-vocabulary words; "
        "cannot honestly claim this language path was exercised."
    )


def main():
    if not CHECKPOINT.is_file() or CHECKPOINT.stat().st_size == 0:
        raise SystemExit(f"FAIL: missing checkpoint: {CHECKPOINT}")

    package = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    assert package.get("format") == "chakra-independent-expert-v1"
    assert package.get("domain") == EXPECTED_DOMAIN
    assert str(package.get("source_run_id")) == EXPECTED_SOURCE_RUN
    assert int(package["V"]) == len(package["vocab"])
    assert int(package["V"]) == int(package["m"]["tok.weight"].shape[0])
    assert int(package["m"]["pat.weight"].shape[0]) == max(len(package["pats"]), 1)

    # Strict state_dict loading is performed inside ExpertRunner.__init__.
    runner = ExpertRunner(CHECKPOINT, "cpu")
    assert runner.domain == EXPECTED_DOMAIN
    assert runner.model.training is False

    # Validate that this expert's graph is internally bounded and non-empty.
    nbrs = runner.graph.get("nbrs") or {}
    cands = runner.graph.get("cands") or {}
    assert nbrs, "Science graph has no neighbour map"
    assert cands, "Science graph has no candidate map"
    V = len(runner.vocab)
    graph_edges = 0
    for mapping in (nbrs, cands):
        for key, values in mapping.items():
            source_id = int(key)
            assert 0 <= source_id < V
            for value in values:
                assert 0 <= int(value) < V
                graph_edges += 1
    assert graph_edges > 0

    hindi_prompt, hindi_known, hindi_oov = choose_prompt(
        runner,
        [
            "परमाणु अणु ऊर्जा रासायनिक अभिक्रिया पदार्थ",
            "रासायनिक अभिक्रिया में ऊर्जा परिवर्तन",
            "परमाणु संरचना और पदार्थ",
            "ऊर्जा पदार्थ अणु परमाणु",
        ],
        "Hindi",
    )
    english_prompt, english_known, english_oov = choose_prompt(
        runner,
        [
            "chemical reaction energy atom molecule",
            "matter energy reaction atomic structure",
            "atom molecule chemical reaction",
            "energy matter molecule reaction",
        ],
        "English",
    )

    results = {}
    for language, prompt in (("Hindi", hindi_prompt), ("English", english_prompt)):
        result = runner.generate(prompt, max_new_tokens=8)
        assert result["domain"] == EXPECTED_DOMAIN
        assert len(result["generated_tokens"]) == 8
        assert all(d["candidate_count"] > 0 for d in result["diagnostics"])
        assert all(torch.isfinite(torch.tensor(d["score"])) for d in result["diagnostics"])
        assert all(0 <= d["pattern_id"] < max(len(runner.patterns), 1)
                   for d in result["diagnostics"])
        results[language] = {
            "prompt": prompt,
            "generated_text": result["text"],
            "known_prompt_tokens": result["known_prompt_tokens"],
            "unknown_prompt_tokens": result["unknown_prompt_tokens"],
            "generated_tokens": len(result["generated_tokens"]),
            "graph_candidate_counts": [d["candidate_count"] for d in result["diagnostics"]],
            "pattern_exact_matches": sum(bool(d["pattern_exact_match"]) for d in result["diagnostics"]),
        }

    report = {
        "status": "PASS",
        "domain": runner.domain,
        "source_run_id": EXPECTED_SOURCE_RUN,
        "checkpoint_bytes": CHECKPOINT.stat().st_size,
        "parameter_count": int(package.get("parameter_count", 0)),
        "vocab_size": len(runner.vocab),
        "pattern_count": len(runner.patterns),
        "graph_neighbour_keys": len(nbrs),
        "graph_candidate_keys": len(cands),
        "graph_edges_checked": graph_edges,
        "strict_state_dict_load": "PASS",
        "graph_id_bounds": "PASS",
        "language_smoke_tests": results,
        "limitations": [
            "Greedy graph-candidate continuation is not an answer-quality benchmark.",
            "Prompt availability is checked against the expert's own vocabulary.",
            "No training, checkpoint mutation, averaging, or quantization was performed.",
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
