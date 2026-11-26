"""Paired pre-training benchmark: native Dhīr baseline vs NeuroCore paths.

This benchmark records deterministic outputs and runtime measurements. It does
not assign an intelligence score; correctness and reasoning quality require
the per-case human rubric in the JSON report.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch

from dhir2b_neurocore import Dhir2BNeuroCore
from neurocore_inference import run_neurocore_inference
from neurocore_tokenizer import TASK_TO_EXPERT

MODEL_ID = "zenteiq-ai/Dhir-2B"
MODEL_REVISION = "7cbe444cd1e45e25197cfa0f683b2cb575858169"

BENCHMARK_CASES: list[dict[str, Any]] = [
    {
        "id": "hi_detailed_01",
        "category": "Hindi detailed response",
        "language_tag": "hi",
        "task": "hindi_indic_docs",
        "prompt": (
            "भारत की भाषाई विविधता और साहित्य का समाज, शिक्षा, इतिहास तथा "
            "राष्ट्रीय एकता पर प्रभाव विस्तार से समझाइए। पाँच मुख्य बिंदु "
            "दीजिए और अंत में निष्कर्ष लिखिए।"
        ),
        "expected_facts": [
            "भारत में अनेक भाषाएँ और साहित्यिक परंपराएँ हैं",
            "भाषा सांस्कृतिक स्मृति और पहचान को संरक्षित करती है",
            "साहित्य सामाजिक विचार और शिक्षा को प्रभावित करता है",
            "भाषाई विविधता और राष्ट्रीय एकता में संतुलन समझाया गया हो",
        ],
        "rubric": [
            "तथ्यात्मक शुद्धता",
            "कम-से-कम पाँच अलग और प्रासंगिक बिंदु",
            "हिंदी की स्पष्टता और विस्तार",
            "निष्कर्ष की संगति",
        ],
    },
    {
        "id": "science_en_01",
        "category": "English scientific explanation",
        "language_tag": "en",
        "task": "technical_academic",
        "prompt": (
            "Explain photosynthesis scientifically. Cover the light-dependent "
            "reactions, ATP and NADPH, the Calvin cycle, the role of carbon "
            "dioxide and water, and where the released oxygen comes from."
        ),
        "expected_facts": [
            "light-dependent reactions occur in the thylakoid membranes",
            "ATP and NADPH provide energy and reducing power",
            "the Calvin cycle fixes carbon dioxide",
            "released oxygen comes from water splitting",
        ],
        "rubric": [
            "Scientific correctness",
            "Correct relationship between light reactions and Calvin cycle",
            "Clear explanation of inputs, outputs and energy transfer",
            "No invented biochemical steps",
        ],
    },
    {
        "id": "math_hi_01",
        "category": "Mathematics",
        "language_tag": "hi",
        "task": "chat",
        "prompt": (
            "एक पानी की टंकी में शुरुआत में 120 लीटर पानी है। उसमें हर मिनट "
            "18 लीटर पानी 15 मिनट तक भरा जाता है। फिर टंकी से 75 लीटर पानी "
            "निकाल दिया जाता है। अंत में कितना पानी बचता है? पूरी गणना "
            "चरण-दर-चरण दिखाइए।"
        ),
        "expected_answer": "315 लीटर",
        "expected_calculation": "120 + (18 × 15) − 75 = 315",
        "rubric": [
            "18 × 15 = 270 की सही गणना",
            "प्रारंभिक मात्रा और भरी गई मात्रा का सही योग",
            "75 लीटर घटाने के बाद 315 लीटर का सही उत्तर",
            "गणना के चरण स्पष्ट हों",
        ],
    },
    {
        "id": "logic_hi_01",
        "category": "Logical reasoning",
        "language_tag": "hi",
        "task": "chat",
        "prompt": (
            "कथन हैं: (1) सभी इंजीनियर गणित पढ़ते हैं। (2) गणित पढ़ने वाला "
            "कोई भी व्यक्ति अशिक्षित नहीं है। (3) रवि एक इंजीनियर है। "
            "क्या इन कथनों से यह निष्कर्ष निकलता है कि रवि अशिक्षित नहीं है? "
            "तर्क के प्रत्येक चरण को स्पष्ट कीजिए।"
        ),
        "expected_answer": "हाँ; दिए गए कथनों के अनुसार रवि गणित पढ़ता है, इसलिए वह अशिक्षित नहीं है।",
        "rubric": [
            "इंजीनियर होने से गणित पढ़ने का निष्कर्ष",
            "गणित पढ़ने से अशिक्षित न होने का निष्कर्ष",
            "निष्कर्ष केवल दिए गए आधारों पर निकाला गया हो",
            "तार्किक चरण स्पष्ट हों",
        ],
    },
    {
        "id": "self_correction_hi_01",
        "category": "Self-correction",
        "language_tag": "hi",
        "task": "self_correction_mnl",
        "prompt": (
            "नीचे दिए कथन में जानबूझकर त्रुटि है: 'सूर्य पृथ्वी के चारों ओर "
            "24 घंटे में घूमता है।' इस कथन की तथ्य-जाँच कीजिए। गलत भाग "
            "पहचानिए, सही वैज्ञानिक स्थिति लिखिए और पृथ्वी के अपने अक्ष पर "
            "घूर्णन तथा सूर्य की परिक्रमा में अंतर स्पष्ट कीजिए।"
        ),
        "expected_facts": [
            "पृथ्वी अपने अक्ष पर लगभग 24 घंटे में घूमती है",
            "पृथ्वी सूर्य की परिक्रमा लगभग 365.25 दिनों में करती है",
            "सूर्य पृथ्वी की परिक्रमा नहीं करता",
        ],
        "rubric": [
            "गलत कथन को स्पष्ट रूप से पहचानना",
            "घूर्णन और परिक्रमण में अंतर",
            "दोनों अवधियों का सही उपयोग",
            "सुधार में नया भ्रम या विरोधाभास न हो",
        ],
    },
    {
        "id": "planning_hi_01",
        "category": "Multi-step planning",
        "language_tag": "hi",
        "task": "coconut_planning",
        "prompt": (
            "किसी गाँव में जल संरक्षण के लिए 90 दिनों की व्यावहारिक योजना "
            "बनाइए। योजना को चरणों में बाँटिए; हर चरण के कार्य, जिम्मेदार "
            "लोग, समय-सीमा, मापने योग्य सफलता-सूचक और जोखिम होने पर "
            "वैकल्पिक कदम बताइए।"
        ),
        "rubric": [
            "चरणों का सही क्रम और निर्भरताएँ",
            "जिम्मेदार व्यक्तियों या समूहों का निर्धारण",
            "90 दिनों के यथार्थवादी milestones",
            "मापने योग्य परिणाम",
            "जोखिम और वैकल्पिक उपाय",
        ],
    },
]


def run_benchmark(
    model: Any,
    tokenizer: Any,
    *,
    max_new_tokens: int = 80,
    do_sample: bool = False,
) -> dict[str, Any]:
    if not 1 <= max_new_tokens <= 128:
        raise ValueError("max_new_tokens must be between 1 and 128")

    rows: list[dict[str, Any]] = []
    mode_totals: dict[str, list[float]] = {}

    for case in BENCHMARK_CASES:
        modes = ["base_dhir"]
        if TASK_TO_EXPERT.get(case["task"]) is not None:
            modes.append("task_expert")
        modes.append("top2")

        case_outputs: dict[str, dict[str, Any]] = {}
        for mode in modes:
            with torch.inference_mode():
                result = run_neurocore_inference(
                    model,
                    tokenizer,
                    case["prompt"],
                    task=case["task"],
                    language_tag=case["language_tag"],
                    mode=mode,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                )
            if result.get("status") != "INFERENCE_COMPLETED":
                raise RuntimeError(f"benchmark inference failed: {case['id']} / {mode}")
            case_outputs[mode] = result
            mode_totals.setdefault(mode, []).append(float(result["tokens_per_second"]))

        baseline_text = case_outputs["base_dhir"]["answer"]
        for mode in modes:
            result = case_outputs[mode]
            row = {
                "case_id": case["id"],
                "category": case["category"],
                "mode": mode,
                "task": case["task"],
                "language_tag": case["language_tag"],
                "suggested_expert_id": result.get("suggested_expert_id"),
                "suggested_expert_name": result.get("suggested_expert_name"),
                "selected_expert_id": result.get("selected_expert_id"),
                "router_status": result.get("router_status"),
                "prompt_tokens": result["prompt_tokens"],
                "generated_tokens": result["generated_tokens"],
                "elapsed_seconds": result["elapsed_seconds"],
                "tokens_per_second": result["tokens_per_second"],
                "answer": result["answer"],
                "exact_text_match_to_base_dhir": (
                    result["answer"] == baseline_text if mode != "base_dhir" else True
                ),
                "rubric": case["rubric"],
                "expected_facts": case.get("expected_facts"),
                "expected_answer": case.get("expected_answer"),
                "expected_calculation": case.get("expected_calculation"),
            }
            rows.append(row)

    return {
        "status": "COMPLETED_UNTRAINED_PAIRED_BASELINE",
        "benchmark_version": 1,
        "base_model_id": MODEL_ID,
        "base_model_revision": MODEL_REVISION,
        "device": "cpu",
        "dtype": str(next(model.backbone.parameters()).dtype),
        "decoder_layers": len(model.adapter_layers),
        "hidden_size": model.hidden_size,
        "vocab_size": model.vocab_size,
        "experts": model.num_experts,
        "experts_per_token": model.adapter_layers[0].top_k,
        "router_status": "UNTRAINED_NOT_QUALITY_VALIDATED",
        "training_performed": False,
        "weights_persisted": False,
        "max_new_tokens": max_new_tokens,
        "sampling": "greedy_deterministic" if not do_sample else "sampling_enabled",
        "cases": len(BENCHMARK_CASES),
        "inference_runs": len(rows),
        "mean_tokens_per_second_by_mode": {
            mode: round(statistics.mean(values), 3)
            for mode, values in sorted(mode_totals.items())
        },
        "scoring": {
            "automatic_quality_score": None,
            "human_review_required": True,
            "note": (
                "This is an untrained baseline and runtime comparison. "
                "Exact text equality is not a quality score. Apply each row's "
                "rubric before claiming reasoning or intelligence gains."
            ),
        },
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not 1 <= args.threads <= 8:
        parser.error("--threads must be between 1 and 8")
    torch.set_num_threads(args.threads)

    model, tokenizer, loading_info = Dhir2BNeuroCore.import_pretrained(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        rank=8,
        dtype=torch.bfloat16,
    )
    clean = not any(
        loading_info.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    )
    if not clean:
        raise RuntimeError("Dhīr checkpoint loading audit was not clean")
    model.eval()

    report = run_benchmark(
        model,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    report["loading_info_clean"] = clean
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
