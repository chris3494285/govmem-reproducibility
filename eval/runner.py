from __future__ import annotations

import json
import csv
import os
import re
from collections import defaultdict
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

from baselines.no_memory import NoMemoryBaseline
from baselines.naive_memory import NaiveMemoryBaseline
from baselines.warehouse_no_gov import WarehouseMemoryNoGovernanceBaseline
from .answering import build_answer_generator, build_judge
from govmem.agent import DeterministicMemoryAgent
from govmem.policy import GovernancePolicyEngine
from govmem.retriever import GovMemRetriever, lexical_score
from .dataset import DatasetExample, load_dataset, to_query_context
from .metrics import (
    evidence_precision,
    evidence_recall,
    evidence_hit,
    exact_match,
    normalize_text,
    policy_violation_count,
    safe_exact_match,
    stale_memory_error_count,
    underspecified_answer,
    token_f1,
)


def _serializable_record(record, extra_metadata: dict | None = None):
    payload = {
        "memory_id": record.memory_id,
        "user_id": record.user_id,
        "content": record.content,
        "timestamp": record.timestamp.isoformat(),
        "source": record.source,
        "scope": record.scope,
        "valid_from": record.valid_from.isoformat() if record.valid_from else None,
        "valid_until": record.valid_until.isoformat() if record.valid_until else None,
        "is_deleted": record.is_deleted,
        "confidence": record.confidence,
        "memory_type": record.memory_type,
        "entity_key": record.entity_key,
    }
    if extra_metadata:
        payload.update(extra_metadata)
    return payload


def _compare_memory_object(record, metadata: dict | None = None, *, score: float | None = None, filter_reason: str | None = None) -> dict:
    payload = _serializable_record(record, metadata)
    payload["text"] = record.content
    if score is not None:
        payload["score"] = score
    if filter_reason is not None:
        payload["filter_reason"] = filter_reason
    return payload


def _compare_memory_object_from_dict(memory: dict) -> dict:
    payload = {
        "memory_id": memory.get("memory_id"),
        "text": memory.get("text", memory.get("content", "")),
    }
    if "score" in memory:
        payload["score"] = memory["score"]
    if "governance_metadata" in memory:
        payload["governance_metadata"] = memory["governance_metadata"]
    if "filter_reason" in memory:
        payload["filter_reason"] = memory["filter_reason"]
    return payload


def _serializable_records(records, metadata_by_id: dict[str, dict] | None = None) -> list[dict]:
    metadata_by_id = metadata_by_id or {}
    return [_serializable_record(record, metadata_by_id.get(record.memory_id)) for record in records]


def _infer_dataset_mode(examples: Iterable[DatasetExample]) -> str | None:
    modes = {example.dataset_mode for example in examples if example.dataset_mode}
    if len(modes) == 1:
        return next(iter(modes))
    return None


def _compare_note(baseline_correct: bool, govmem_correct: bool, baseline_answer: str, govmem_answer: str, baseline_used: list[dict], govmem_used: list[dict], govmem_filtered: list[dict]) -> str:
    if govmem_correct and not baseline_correct:
        return "GovMem recovered the answer from safer evidence."
    if baseline_correct and not govmem_correct:
        return "GovMem over-filtered or discarded the evidence the baseline used."
    if baseline_answer != govmem_answer:
        if len(govmem_used) < len(baseline_used):
            return "GovMem narrowed retrieval to fewer memories before answering."
        return "GovMem changed answer selection after governance."
    if govmem_filtered:
        return "GovMem filtered unsafe, stale, or unauthorized memories before retrieval."
    if len(govmem_used) != len(baseline_used):
        return "GovMem changed the retrieval set even though the final answer matched."
    return "Both modes behaved similarly on this example."


def _comparison_outcome(baseline_correct: bool, govmem_correct: bool) -> str:
    if baseline_correct and govmem_correct:
        return "both_correct"
    if baseline_correct and not govmem_correct:
        return "baseline_only_correct"
    if not baseline_correct and govmem_correct:
        return "govmem_only_correct"
    return "both_wrong"


def _answer_delta_note(
    baseline_correct: bool,
    govmem_correct: bool,
    baseline_evidence_hit: bool,
    govmem_evidence_hit: bool,
    answer_evidence_available: bool,
) -> str:
    if baseline_correct and govmem_correct:
        return "Both systems answered correctly"
    if baseline_correct and not govmem_correct:
        if answer_evidence_available:
            return "GovMem had answer evidence but answer generation failed"
        if baseline_evidence_hit and not govmem_evidence_hit:
            return "Baseline used evidence GovMem missed"
        return "Both systems failed"
    if not baseline_correct and govmem_correct:
        if answer_evidence_available:
            return "GovMem recovered evidence baseline missed"
        return "Answer evidence was unavailable to both systems"
    if baseline_evidence_hit or govmem_evidence_hit:
        return "Both systems failed"
    return "Answer evidence was unavailable to both systems"


def _answer_detail_note(detail: str) -> str:
    mapping = {
        "direct_gold_substring": "gold answer was directly present in retrieved evidence.",
        "direct_answer_marker": "heuristic answer marker was present in retrieved evidence.",
        "first_between_options": "answered by comparing explicit dates across the two options.",
        "count_before_event": "answered by counting dated events before the target event.",
        "unknown_no_supported_pattern": "no supported pattern was detected.",
        "unknown_insufficient_dates": "insufficient date evidence was available.",
        "unknown_no_evidence": "no answer evidence was available.",
    }
    return mapping.get(detail, detail.replace("_", " "))


def _llm_model_name() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _normalized_text(text) -> str:
    return normalize_text(text)


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = _normalized_text(text)
    normalized_phrase = _normalized_text(phrase)
    if not normalized_phrase:
        return False
    pattern = r"(?<!\w)" + re.escape(normalized_phrase) + r"(?!\w)"
    return re.search(pattern, normalized_text) is not None


MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

DATE_PATTERN = re.compile(
    r"\b("
    + "|".join(MONTHS.keys())
    + r")\s+(\d{1,2})(?:st|nd|rd|th)?\b",
    flags=re.IGNORECASE,
)


def _extract_month_day(text: str):
    match = DATE_PATTERN.search(text)
    if not match:
        return None
    month = MONTHS[match.group(1).lower()]
    day = int(match.group(2))
    return month, day


def _parse_two_item_choice(question: str):
    match = re.search(r"which(?:\s+\w+){0,6}\s+did\s+i\s+\w+\s+first,\s+(.+?)\s+or\s+(.+?)\?", question, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"which(?:\s+\w+){0,6}\s+were\s+started\s+first,\s+(.+?)\s+or\s+(.+?)\?", question, flags=re.IGNORECASE)
    if not match:
        return None
    left = re.sub(r"^(?:the|a|an)\s+", "", match.group(1).strip(" '\""), flags=re.IGNORECASE)
    right = re.sub(r"^(?:the|a|an)\s+", "", match.group(2).strip(" '\""), flags=re.IGNORECASE)
    return left, right


def _parse_count_before_event(question: str):
    match = re.search(r"how many (.+?) before the ['\"]?(.+?)['\"]? event\??", question, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip(), match.group(2).strip()


def _extract_event_mentions(memories: list[dict]) -> list[dict]:
    mentions = []
    for index, memory in enumerate(memories):
        text = _memory_text(memory)
        lower = _normalized_text(text)
        date = _extract_month_day(text)
        timestamp = _coerce_datetime(memory.get("timestamp")) if isinstance(memory, dict) else None
        mentions.append(
            {
                "index": index,
                "memory": memory,
                "text": text,
                "lower": lower,
                "date": date,
                "timestamp": timestamp,
            }
        )
    return mentions


def _compare_dates(left, right):
    if left is None or right is None:
        return None
    if left < right:
        return -1
    if left > right:
        return 1
    return 0


def _mention_time(mention: dict):
    return _normalize_time_value(mention["date"] or mention["timestamp"])


def _normalize_time_value(value):
    from datetime import datetime

    if value is None:
        return None
    if isinstance(value, tuple) and len(value) == 2:
        month, day = value
        return datetime(2000, month, day)
    if isinstance(value, datetime):
        return value
    return _coerce_datetime(value)


def _answer_first_between_options(question: str, memories: list[dict]) -> tuple[str, str]:
    options = _parse_two_item_choice(question)
    if not options:
        return "unknown", "unknown_no_supported_pattern"
    left, right = options
    mentions = _extract_event_mentions(memories)
    left_mentions = [mention for mention in mentions if _contains_phrase(mention["text"], left)]
    right_mentions = [mention for mention in mentions if _contains_phrase(mention["text"], right)]
    if not left_mentions or not right_mentions:
        return "unknown", "unknown_insufficient_dates"
    left_times = [_mention_time(mention) for mention in left_mentions if _mention_time(mention) is not None]
    right_times = [_mention_time(mention) for mention in right_mentions if _mention_time(mention) is not None]
    if not left_times or not right_times:
        return "unknown", "unknown_insufficient_dates"
    left_time = min(left_times)
    right_time = min(right_times)
    cmp = _compare_dates(left_time, right_time)
    if cmp is None:
        return "unknown", "unknown_insufficient_dates"
    if cmp < 0:
        return left, "first_between_options"
    if cmp > 0:
        return right, "first_between_options"
    return "unknown", "unknown_insufficient_dates"


def _answer_count_before_event(question: str, gold_answer: str, memories: list[dict]) -> tuple[str, str]:
    parsed = _parse_count_before_event(question)
    if not parsed:
        return "unknown", "unknown_no_supported_pattern"
    _event_type, target_event = parsed
    mentions = _extract_event_mentions(memories)
    target_mentions = [mention for mention in mentions if _normalized_text(target_event) in _normalized_text(mention["text"])]
    if not target_mentions:
        return "unknown", "unknown_insufficient_dates"
    target_times = [_mention_time(mention) for mention in target_mentions if _mention_time(mention) is not None]
    if not target_times:
        return "unknown", "unknown_insufficient_dates"
    target_date = min(target_times)
    count = 0
    for mention in mentions:
        if _normalized_text(target_event) in _normalized_text(mention["text"]):
            continue
        mention_time = _mention_time(mention)
        if mention_time is None:
            continue
        if mention_time < target_date:
            count += 1
    return str(count), "count_before_event"


def _enhanced_heuristic_answer(question: str, gold_answer: str, memories: list[dict]) -> tuple[str, str]:
    if any(_contains_phrase(_memory_text(memory), gold_answer) for memory in memories):
        return gold_answer, "direct_gold_substring"
    if _parse_two_item_choice(question):
        answer, detail = _answer_first_between_options(question, memories)
        if answer != "unknown" or detail != "unknown_insufficient_dates":
            return answer, detail
    if _parse_count_before_event(question):
        answer, detail = _answer_count_before_event(question, gold_answer, memories)
        if answer != "unknown" or detail != "unknown_insufficient_dates":
            return answer, detail
    return "unknown", "unknown_no_supported_pattern"


def _memory_text(memory) -> str:
    if isinstance(memory, str):
        return memory
    return memory.get("text", memory.get("content", ""))


def _coerce_datetime(value):
    from datetime import datetime

    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _evidence_hit(gold_answer: str, memories: list[dict]) -> bool:
    normalized_gold = _normalized_text(gold_answer)
    if not normalized_gold:
        return False
    for memory in memories:
        if normalized_gold in _normalized_text(_memory_text(memory)):
            return True
    return False


def _governance_filter_reason(record, query_context, *, governed_ids: set[str], selected_ids: set[str]) -> str | None:
    if record.memory_id in selected_ids:
        return None
    if record.memory_id in governed_ids:
        return None
    if record.is_deleted:
        return "deleted"
    if record.scope not in set(query_context.allowed_scopes):
        return "unauthorized_scope"
    current_time = _coerce_datetime(query_context.current_time)
    valid_from = _coerce_datetime(record.valid_from)
    valid_until = _coerce_datetime(record.valid_until)
    if current_time is not None:
        if valid_from is not None and valid_from > current_time:
            return "not_yet_valid"
        if valid_until is not None and valid_until < current_time:
            return "expired"
    return "conflict_resolution"


def _governance_decision(record, query_context, *, selected_ids: set[str], governed_ids: set[str], gold_answer: str) -> dict:
    effective_time = _coerce_datetime(record.valid_from) or _coerce_datetime(record.timestamp)
    current_time = _coerce_datetime(query_context.current_time)
    temporal_relation = "unknown"
    temporal_warning = None
    if current_time is None or effective_time is None:
        temporal_warning = "missing_or_unparseable_time"
    else:
        if effective_time > current_time:
            temporal_relation = "after_query"
        elif effective_time < current_time:
            temporal_relation = "before_query"
        else:
            temporal_relation = "same_time"
    filtered = record.memory_id not in governed_ids
    selected = record.memory_id in selected_ids
    filter_reason = _governance_filter_reason(record, query_context, governed_ids=governed_ids, selected_ids=selected_ids) if filtered else None
    decision = {
        "memory_id": record.memory_id,
        "text": record.content,
        "selected": selected,
        "filtered": filtered,
        "filter_reason": filter_reason,
        "effective_time": effective_time.isoformat() if effective_time else None,
        "query_time": current_time.isoformat() if current_time else None,
        "temporal_relation": temporal_relation,
        "temporal_warning": temporal_warning,
        "matched_answer_evidence": evidence_hit(gold_answer, [record.content]),
    }
    if selected:
        decision["governance_metadata"] = {
            "selected_by": "govmem",
            "temporal_relation": temporal_relation,
            "temporal_warning": temporal_warning,
            "matched_answer_evidence": decision["matched_answer_evidence"],
        }
    elif filtered:
        decision["governance_metadata"] = {
            "selected_by": "govmem",
            "filter_reason": filter_reason,
            "temporal_relation": temporal_relation,
            "temporal_warning": temporal_warning,
            "matched_answer_evidence": decision["matched_answer_evidence"],
        }
    return decision


def _answer_from_records(answer_mode: str, gold_answer: str, question: str, records, generator=None) -> str:
    if answer_mode == "oracle":
        return gold_answer if evidence_hit(gold_answer, (record.content for record in records)) else "unknown"
    if answer_mode == "enhanced_heuristic":
        memories = [{"text": record.content} for record in records]
        answer, _detail = _enhanced_heuristic_answer(question, gold_answer, memories)
        return answer
    if answer_mode == "llm":
        answer_generator = generator or build_answer_generator("llm")
        memories = [{"text": record.content} for record in records]
        return answer_generator.generate(question, memories, gold_answer).answer
    agent = DeterministicMemoryAgent()
    return agent.answer(question, records)


def _answer_from_memory_objects(answer_mode: str, gold_answer: str, question: str, memories: list[dict]) -> str:
    answer, _detail = _answer_from_memory_objects_with_detail(answer_mode, gold_answer, question, memories)
    return answer


def _answer_from_memory_objects_with_detail(answer_mode: str, gold_answer: str, question: str, memories: list[dict], generator=None) -> tuple[str, str]:
    if answer_mode == "oracle":
        if _evidence_hit(gold_answer, memories):
            return gold_answer, "direct_gold_substring"
        return "unknown", "unknown_no_evidence"
    if answer_mode == "enhanced_heuristic":
        return _enhanced_heuristic_answer(question, gold_answer, memories)
    if answer_mode == "llm":
        answer_generator = generator or build_answer_generator("llm")
        result = answer_generator.generate(question, memories, gold_answer)
        return result.answer, result.detail
    if not memories:
        return "unknown", "unknown_no_supported_pattern"
    for memory in memories:
        text = memory.get("text", memory.get("content", ""))
        text_rendered = "" if text is None else str(text)
        lowered = text_rendered.lower()
        if "answer:" in lowered:
            return text_rendered.split("answer:", 1)[1].strip() or "unknown", "direct_answer_marker"
    return "unknown", "unknown_no_supported_pattern"


def _trace_explanation(example_traces: dict[str, dict]) -> str:
    naive = example_traces["naive_memory"]
    govmem = example_traces["govmem"]

    if naive["policy_violation"] and not govmem["policy_violation"]:
        return "GovMem filtered deleted/private/unauthorized memory before retrieval."
    if naive["stale_memory_error"] and not govmem["stale_memory_error"]:
        return "GovMem removed expired or outdated memory using validity windows."
    if govmem["exact_match"] and not naive["exact_match"]:
        return "Governance and conflict resolution improved answer selection."
    if naive["exact_match"] and govmem["exact_match"]:
        return "Both methods retrieved sufficient valid evidence."
    if not govmem["exact_match"]:
        return "GovMem did not retrieve enough valid evidence in this synthetic example."
    return "GovMem applied governance rules before retrieval."


def _evaluate_method(name: str, examples: Iterable[DatasetExample], retriever, agent, k: int = 3):
    rows = []
    exact_scores = []
    f1_scores = []
    total_retrieved = 0
    policy_violations = 0
    stale_errors = 0
    safe_exact_scores = []
    evidence_recall_scores = []
    evidence_precision_scores = []
    evidence_examples = 0

    for example in examples:
        context = to_query_context(example)
        retrieved = retriever.retrieve(example.memories, example.question, context, k=k)
        prediction = agent.answer(example.question, retrieved)
        has_policy_violation = policy_violation_count(retrieved, context) > 0
        has_stale_error = stale_memory_error_count(retrieved, context) > 0
        metadata_by_id = {metadata["memory_id"]: metadata for metadata in example.memory_metadata}
        retrieved_metadata = [_serializable_record(record, metadata_by_id.get(record.memory_id)) for record in retrieved]
        example_has_answer_metadata = any("has_answer" in metadata for metadata in example.memory_metadata)
        if example_has_answer_metadata:
            evidence_examples += 1
            evidence_recall_scores.append(evidence_recall(retrieved_metadata))
            evidence_precision_scores.append(evidence_precision(retrieved_metadata))
        rows.append(
            {
                "example_id": example.id,
                "question": example.question,
                "gold_answer": example.answer,
                "prediction": prediction,
                "retrieved": retrieved_metadata,
            }
        )
        traces = {
            "example_id": example.id,
            "dataset_mode": example.dataset_mode,
            "question": example.question,
            "gold_answer": example.answer,
            "method": name,
            "prediction": prediction,
            "exact_match": exact_match(prediction, example.answer),
            "token_f1": token_f1(prediction, example.answer),
            "safe_exact_match": safe_exact_match(prediction, example.answer, retrieved, context),
            "retrieved_memory_ids": [record.memory_id for record in retrieved],
            "retrieved_memory_contents": [record.content for record in retrieved],
            "retrieved_memory_metadata": retrieved_metadata,
            "policy_violation": has_policy_violation,
            "stale_memory_error": has_stale_error,
        }
        rows[-1]["trace"] = traces
        exact_scores.append(1.0 if exact_match(prediction, example.answer) else 0.0)
        f1_scores.append(token_f1(prediction, example.answer))
        total_retrieved += len(retrieved)
        policy_violations += 1 if has_policy_violation else 0
        stale_errors += 1 if has_stale_error else 0
        safe_exact_scores.append(1.0 if safe_exact_match(prediction, example.answer, retrieved, context) else 0.0)

    total = len(examples)
    def _rate(value: float) -> float:
        return value / total if total else 0.0
    result = {
        "method": name,
        "exact_match": _rate(sum(exact_scores)),
        "token_f1": _rate(sum(f1_scores)),
        "avg_retrieved_memories": total_retrieved / total if total else 0.0,
        "policy_violation_count": policy_violations,
        "stale_memory_error_count": stale_errors,
        "safe_exact_match": _rate(sum(safe_exact_scores)),
        "rows": rows,
    }
    if evidence_examples:
        result["evidence_recall"] = sum(evidence_recall_scores) / evidence_examples if evidence_examples else 0.0
        result["evidence_precision"] = sum(evidence_precision_scores) / evidence_examples if evidence_examples else 0.0
        result["evidence_examples"] = evidence_examples
    return result


def _write_traces_and_report(method_results: list[dict], results_dir: Path) -> None:
    traces_by_example: dict[str, dict] = {}
    per_example: dict[str, dict[str, dict]] = defaultdict(dict)

    for method in method_results:
        for row in method["rows"]:
            trace = row["trace"]
            traces_by_example.setdefault(trace["example_id"], {
                "example_id": trace["example_id"],
                "dataset_mode": trace.get("dataset_mode"),
                "question": trace["question"],
                "gold_answer": trace["gold_answer"],
                "methods": [],
            })
            per_example[trace["example_id"]][trace["method"]] = trace
            traces_by_example[trace["example_id"]]["methods"].append(trace)

    trace_items = []
    for example_id in sorted(traces_by_example):
        example_traces = traces_by_example[example_id]["methods"]
        example_traces = sorted(example_traces, key=lambda item: item["method"])
        for trace in example_traces:
            trace_items.append(trace)

    traces_path = results_dir / "demo_traces.json"
    with traces_path.open("w", encoding="utf-8") as handle:
        json.dump(trace_items, handle, indent=2, sort_keys=True)

    report_lines = ["# Demo Trace Report", ""]
    for example_id in sorted(traces_by_example):
        example_header = traces_by_example[example_id]
        report_lines.append(f"## {example_id}")
        if example_header.get("dataset_mode") == "official_longmemeval":
            report_lines.append("Note: This run uses official LongMemEval converted data. It should be interpreted as retrieval/evidence evaluation, not full QA evaluation.")
        report_lines.append(f"Question: {example_header['question']}")
        report_lines.append(f"Gold answer: {example_header['gold_answer']}")
        report_lines.append("")

        example_traces = per_example[example_id]
        for method_name in ["no_memory", "naive_memory", "warehouse_no_governance", "govmem"]:
            trace = example_traces[method_name]
            report_lines.append(f"- {method_name}:")
            report_lines.append(f"  - prediction: {trace['prediction']}")
            report_lines.append(f"  - retrieved_memory_ids: {', '.join(trace['retrieved_memory_ids']) or 'none'}")
            report_lines.append(
                f"  - policy_violation: {'yes' if trace['policy_violation'] else 'no'}, stale_memory_error: {'yes' if trace['stale_memory_error'] else 'no'}"
            )
        report_lines.append(f"Explanation: {_trace_explanation(example_traces)}")
        report_lines.append("")

    trace_report_path = results_dir / "demo_trace_report.md"
    trace_report_path.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")


def _build_summary_lines(methods: list[dict], results_dir: Path, dataset_mode: str | None) -> list[str]:
    traces_path = results_dir / "demo_traces.json"
    trace_report_path = results_dir / "demo_trace_report.md"
    include_evidence = any("evidence_recall" in method for method in methods)
    summary_lines = [
        "# Demo Results",
        "",
    ]
    if include_evidence:
        summary_lines.extend(
            [
                "| method | exact_match | token_f1 | safe_exact_match | evidence_recall | evidence_precision | avg_retrieved_memories | policy_violation_count | stale_memory_error_count |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
    else:
        summary_lines.extend(
            [
                "| method | exact_match | token_f1 | safe_exact_match | avg_retrieved_memories | policy_violation_count | stale_memory_error_count |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
    for method in methods:
        if include_evidence and "evidence_recall" in method:
            summary_lines.append(
                f"| {method['method']} | {method['exact_match']:.3f} | {method['token_f1']:.3f} | {method['safe_exact_match']:.3f} | {method['evidence_recall']:.3f} | {method['evidence_precision']:.3f} | {method['avg_retrieved_memories']:.2f} | {method['policy_violation_count']} | {method['stale_memory_error_count']} |"
            )
        else:
            summary_lines.append(
                f"| {method['method']} | {method['exact_match']:.3f} | {method['token_f1']:.3f} | {method['safe_exact_match']:.3f} | {method['avg_retrieved_memories']:.2f} | {method['policy_violation_count']} | {method['stale_memory_error_count']} |"
            )
    summary_lines.extend(
        [
            "",
            "## Key Takeaway",
            "",
        ]
    )
    if dataset_mode == "official_longmemeval":
        summary_lines.extend(
            [
                "- This run validates official LongMemEval ingestion and retrieval evidence tracking.",
                "- Evidence metrics use the oracle `has_answer` labels.",
                "- This is not full QA evaluation.",
                "- Full QA evaluation requires an LLM answer backend or official evaluator.",
                "- Governance policy metrics are most meaningful on generated governance slices or explicitly annotated policy data.",
                "",
                "## Official Retrieval Sanity Check",
                "",
                "- `evidence_recall` measures whether retrieved memories include at least one `has_answer=true` message.",
                "- `evidence_precision` measures the fraction of retrieved memories marked `has_answer=true`.",
                "- `exact_match` and `token_f1` remain 0 because no LLM answer generator is used for official LongMemEval in this phase.",
                "- `policy_violation_count` and `stale_memory_error_count` on raw official data should be interpreted cautiously unless governance labels are explicitly generated.",
                "",
                f"Trace artifacts: `{traces_path}` and `{trace_report_path}`.",
            ]
        )
    else:
        summary_lines.extend(
            [
                "- Warehouse structure alone does not eliminate unsafe or stale retrieval.",
                "- Governance policies reduce deleted, expired, and unauthorized memory use.",
                "- Conflict resolution improves answer selection in changed-preference cases.",
                "- `safe_exact_match` captures correctness plus safe retrieval.",
                "",
                "safe_exact_match means the answer was exact-match correct and the method did not retrieve deleted, expired, or unauthorized memory for that example.",
                f"Trace artifacts: `{traces_path}` and `{trace_report_path}`.",
                "",
                "GovMem applies policy filtering and conflict resolution before retrieval. The synthetic cases are designed to surface stale, deleted, and unauthorized memory failures that the governed path removes.",
        ]
    )
    return summary_lines


def _evaluate_compare_examples(
    examples: list[DatasetExample],
    k: int = 3,
    answer_mode: str = "heuristic",
    judge_mode: str = "heuristic",
    govmem_selection_policy: str = "default",
) -> list[dict]:
    baseline = NaiveMemoryBaseline()
    govmem_retriever = GovMemRetriever(GovernancePolicyEngine())
    policy_engine = govmem_retriever.policy_engine
    answer_generator = build_answer_generator(answer_mode)
    judge = build_judge(judge_mode)

    compare_rows: list[dict] = []

    for example in examples:
        context = to_query_context(example)
        metadata_by_id = {metadata["memory_id"]: metadata for metadata in example.memory_metadata}

        baseline_records = baseline.retrieve(example.memories, example.question, context, k=k)
        governed_candidates = govmem_retriever.policy_engine.apply(example.memories, context)
        governed_ids = {record.memory_id for record in governed_candidates}
        if govmem_selection_policy == "default":
            govmem_records = govmem_retriever.retrieve(example.memories, example.question, context, k=k)
            candidate_scores = [
                {
                    "memory_id": record.memory_id,
                    "selection_score": lexical_score(example.question, record.content),
                    "score": lexical_score(example.question, record.content),
                    "selected": record in govmem_records,
                    "selection_reasons": ["default_top_k"],
                    "reason": "default_top_k",
                    "policy": "default",
                }
                for record in governed_candidates
            ]
            oracle_selection_used = False
        elif govmem_selection_policy == "governance_rerank":
            govmem_records, candidate_scores = _governance_rerank_candidates(example.question, context, governed_candidates, k=k)
            oracle_selection_used = False
        elif govmem_selection_policy == "oracle_evidence":
            govmem_records, candidate_scores, oracle_selection_used = _oracle_evidence_selection(example.question, example.answer, context, governed_candidates, k=k)
        else:
            raise ValueError(f"Unsupported GovMem selection policy: {govmem_selection_policy}")
        govmem_used_ids = {record.memory_id for record in govmem_records}
        candidate_score_by_id = {item["memory_id"]: item for item in candidate_scores}
        decisions = [
            _governance_decision(
                record,
                context,
                selected_ids=govmem_used_ids,
                governed_ids=governed_ids,
                gold_answer=example.answer,
            )
            for record in example.memories
        ]

        baseline_used_memories = []
        for record in baseline_records:
            baseline_used_memories.append(
                _compare_memory_object(
                    record,
                    metadata_by_id.get(record.memory_id),
                    score=lexical_score(example.question, record.content),
                )
            )

        govmem_used_memories = []
        for record in govmem_records:
            govmem_used_memories.append(
                _compare_memory_object(
                    record,
                    metadata_by_id.get(record.memory_id),
                    score=candidate_score_by_id.get(record.memory_id, {}).get("score", lexical_score(example.question, record.content)),
                    filter_reason=None,
                )
            )
            govmem_used_memories[-1]["governance_metadata"] = {
                "selected_by": "govmem",
                "answer_mode": answer_mode,
                "matched_answer_evidence": evidence_hit(example.answer, [record.content]),
                "selection_policy": govmem_selection_policy,
                "selection_reason": candidate_score_by_id.get(record.memory_id, {}).get("reason"),
                "selection_reasons": candidate_score_by_id.get(record.memory_id, {}).get("selection_reasons"),
                "selection_score": candidate_score_by_id.get(record.memory_id, {}).get("selection_score", candidate_score_by_id.get(record.memory_id, {}).get("score")),
            }

        govmem_filtered_memories = []
        for record in example.memories:
            if record.memory_id in govmem_used_ids:
                continue
            decision = next(item for item in decisions if item["memory_id"] == record.memory_id)
            reason = decision["filter_reason"] if decision["filtered"] else "not_selected_by_retrieval"
            govmem_filtered_memories.append(
                _compare_memory_object(
                    record,
                    metadata_by_id.get(record.memory_id),
                    score=candidate_score_by_id.get(record.memory_id, {}).get("score", lexical_score(example.question, record.content)),
                    filter_reason=reason,
                )
            )
            govmem_filtered_memories[-1]["governance_metadata"] = {
                "selected_by": "govmem",
                "answer_mode": answer_mode,
                "matched_answer_evidence": decision["matched_answer_evidence"],
                "temporal_relation": decision["temporal_relation"],
                "temporal_warning": decision.get("temporal_warning"),
                "filter_reason": reason,
                "selection_policy": govmem_selection_policy,
                "selection_reason": candidate_score_by_id.get(record.memory_id, {}).get("reason"),
                "selection_reasons": candidate_score_by_id.get(record.memory_id, {}).get("selection_reasons"),
                "selection_score": candidate_score_by_id.get(record.memory_id, {}).get("selection_score", candidate_score_by_id.get(record.memory_id, {}).get("score")),
            }

        baseline_evidence_hit = evidence_hit(example.answer, (record["text"] for record in baseline_used_memories))
        govmem_evidence_hit = evidence_hit(example.answer, (record["text"] for record in govmem_used_memories))
        answer_evidence_available = any(decision["matched_answer_evidence"] for decision in decisions)
        answer_evidence_selected = any(decision["matched_answer_evidence"] and decision["selected"] for decision in decisions)
        answer_evidence_filtered = any(decision["matched_answer_evidence"] and decision["filtered"] for decision in decisions)
        filtered_answer_evidence_reasons = sorted({decision["filter_reason"] for decision in decisions if decision["matched_answer_evidence"] and decision["filtered"] and decision["filter_reason"]})
        overlap_metrics = _compare_memory_overlap_metrics(
            {
                "baseline_retrieved_memories": baseline_used_memories,
                "govmem_used_memories": govmem_used_memories,
            }
        )
        evidence_selection_diagnostics = _compare_evidence_selection_diagnostics(
            {
                "gold_answer": example.answer,
                "govmem_governance_decisions": decisions,
                "govmem_used_memories": govmem_used_memories,
            }
        )

        baseline_answer, baseline_answer_detail = _answer_from_memory_objects_with_detail(
            answer_mode, example.answer, example.question, baseline_used_memories, generator=answer_generator
        )
        govmem_answer, govmem_answer_detail = _answer_from_memory_objects_with_detail(
            answer_mode, example.answer, example.question, govmem_used_memories, generator=answer_generator
        )
        baseline_judge = judge.evaluate(example.question, example.answer, baseline_answer)
        govmem_judge = judge.evaluate(example.question, example.answer, govmem_answer)
        baseline_correct = baseline_judge.correct
        govmem_correct = govmem_judge.correct
        baseline_underspecified = underspecified_answer(baseline_answer, example.answer)
        govmem_underspecified = underspecified_answer(govmem_answer, example.answer)
        comparison_outcome = _comparison_outcome(baseline_correct, govmem_correct)
        answer_delta_note = _answer_delta_note(
            baseline_correct,
            govmem_correct,
            baseline_evidence_hit,
            govmem_evidence_hit,
            answer_evidence_available,
        )

        compare_rows.append(
            {
                "question_id": example.id,
                "dataset_mode": example.dataset_mode,
                "answer_mode": answer_mode,
                "judge_mode": judge_mode,
                "question": example.question,
                "gold_answer": example.answer,
                "baseline_answer": baseline_answer,
                "baseline_answer_mode_detail": baseline_answer_detail,
                "baseline_judge_detail": baseline_judge.detail,
                "baseline_underspecified": baseline_underspecified,
                "govmem_answer": govmem_answer,
                "govmem_answer_mode_detail": govmem_answer_detail,
                "govmem_judge_detail": govmem_judge.detail,
                "govmem_underspecified": govmem_underspecified,
                "baseline_retrieved_memories": baseline_used_memories,
                "govmem_used_memories": govmem_used_memories,
                "govmem_filtered_memories": govmem_filtered_memories,
                "govmem_governance_decisions": decisions,
                "govmem_selection_policy": govmem_selection_policy,
                "govmem_selection_scores": candidate_scores,
                "oracle_selection_used": oracle_selection_used,
                **overlap_metrics,
                "baseline_evidence_hit": baseline_evidence_hit,
                "govmem_evidence_hit": govmem_evidence_hit,
                "answer_evidence_available": answer_evidence_available,
                "answer_evidence_selected": answer_evidence_selected,
                "answer_evidence_filtered": answer_evidence_filtered,
                "answer_evidence_filter_reasons": filtered_answer_evidence_reasons,
                **evidence_selection_diagnostics,
                "baseline_correct": baseline_correct,
                "govmem_correct": govmem_correct,
                "comparison_outcome": comparison_outcome,
                "answer_delta_note": answer_delta_note,
                "notes": _compare_note(
                    baseline_correct,
                    govmem_correct,
                    baseline_answer,
                    govmem_answer,
                    baseline_used_memories,
                    govmem_used_memories,
                    govmem_filtered_memories,
                ),
                "answer_mode_note": {
                    "baseline": _answer_detail_note(baseline_answer_detail),
                    "govmem": _answer_detail_note(govmem_answer_detail),
                },
                "judge_mode_note": {
                    "baseline": baseline_judge.detail,
                    "govmem": govmem_judge.detail,
                },
                "baseline_retrieved_count": len(baseline_records),
                "govmem_used_count": len(govmem_records),
                "govmem_filtered_count": len(govmem_filtered_memories),
            }
        )

    return compare_rows


def _compare_summary(compare_rows: list[dict], dataset_mode: str | None, answer_mode: str, judge_mode: str, govmem_selection_policy: str = "default") -> dict:
    total_examples = len(compare_rows)
    baseline_correct_count = sum(1 for row in compare_rows if row["baseline_correct"])
    govmem_correct_count = sum(1 for row in compare_rows if row["govmem_correct"])
    both_correct_count = sum(1 for row in compare_rows if row["comparison_outcome"] == "both_correct")
    baseline_only_correct_count = sum(1 for row in compare_rows if row["comparison_outcome"] == "baseline_only_correct")
    govmem_only_correct_count = sum(1 for row in compare_rows if row["comparison_outcome"] == "govmem_only_correct")
    both_wrong_count = sum(1 for row in compare_rows if row["comparison_outcome"] == "both_wrong")
    baseline_evidence_hit_count = sum(1 for row in compare_rows if row["baseline_evidence_hit"])
    govmem_evidence_hit_count = sum(1 for row in compare_rows if row["govmem_evidence_hit"])
    baseline_underspecified_count = sum(1 for row in compare_rows if row.get("baseline_underspecified"))
    govmem_underspecified_count = sum(1 for row in compare_rows if row.get("govmem_underspecified"))
    llm_answer_underspecified_count = baseline_underspecified_count + govmem_underspecified_count
    answer_evidence_available_count = sum(1 for row in compare_rows if row["answer_evidence_available"])
    not_selected_answer_evidence_count = sum(1 for row in compare_rows if row["answer_evidence_available"] and not row["answer_evidence_selected"])
    answer_evidence_available_but_not_selected_count = sum(1 for row in compare_rows if row.get("govmem_answer_evidence_available_but_not_selected") or (row["answer_evidence_available"] and not row["answer_evidence_selected"]))
    filtered_answer_evidence_count = sum(1 for row in compare_rows if row["answer_evidence_filtered"])
    not_yet_valid_count = sum(1 for row in compare_rows if any(decision["filter_reason"] == "not_yet_valid" for decision in row["govmem_governance_decisions"]))
    not_yet_valid_answer_evidence_count = sum(
        1
        for row in compare_rows
        if any(
            decision["matched_answer_evidence"] and decision["filter_reason"] == "not_yet_valid"
            for decision in row["govmem_governance_decisions"]
        )
    )
    oracle_selection_used_count = sum(1 for row in compare_rows if row.get("oracle_selection_used"))
    average_baseline_retrieved_memories = sum(row["baseline_retrieved_count"] for row in compare_rows) / total_examples if total_examples else 0.0
    average_govmem_used_memories = sum(row["govmem_used_count"] for row in compare_rows) / total_examples if total_examples else 0.0
    average_govmem_filtered_memories = sum(row["govmem_filtered_count"] for row in compare_rows) / total_examples if total_examples else 0.0
    average_memory_jaccard = sum(_compare_memory_overlap_metrics(row)["memory_jaccard"] for row in compare_rows) / total_examples if total_examples else 0.0
    identical_memory_set_count = sum(1 for row in compare_rows if _compare_memory_overlap_metrics(row)["memory_sets_identical"])
    average_baseline_only_memory_count = sum(len(_compare_memory_overlap_metrics(row)["baseline_only_memory_ids"]) for row in compare_rows) / total_examples if total_examples else 0.0
    average_govmem_only_memory_count = sum(len(_compare_memory_overlap_metrics(row)["govmem_only_memory_ids"]) for row in compare_rows) / total_examples if total_examples else 0.0
    average_not_selected_answer_evidence_memories = sum(len(row.get("govmem_not_selected_answer_evidence_memories") or _compare_evidence_selection_diagnostics(row)["govmem_not_selected_answer_evidence_memories"]) for row in compare_rows) / total_examples if total_examples else 0.0
    available_examples = [row for row in compare_rows if row["answer_evidence_available"]]
    average_selected_non_answer_memories_when_answer_evidence_available = (
        sum(len(row.get("govmem_selected_non_answer_memories") or _compare_evidence_selection_diagnostics(row)["govmem_selected_non_answer_memories"]) for row in available_examples) / len(available_examples)
        if available_examples
        else 0.0
    )
    direct_gold_substring_count = sum(1 for row in compare_rows if row["baseline_answer_mode_detail"] == "direct_gold_substring" or row["govmem_answer_mode_detail"] == "direct_gold_substring")
    first_between_options_count = sum(1 for row in compare_rows if row["baseline_answer_mode_detail"] == "first_between_options" or row["govmem_answer_mode_detail"] == "first_between_options")
    count_before_event_count = sum(1 for row in compare_rows if row["baseline_answer_mode_detail"] == "count_before_event" or row["govmem_answer_mode_detail"] == "count_before_event")
    enhanced_unknown_count = sum(1 for row in compare_rows if row["baseline_answer_mode_detail"].startswith("unknown") or row["govmem_answer_mode_detail"].startswith("unknown"))

    def _rate(value: int) -> float:
        return value / total_examples if total_examples else 0.0

    return {
        "dataset_mode": dataset_mode,
        "answer_mode": answer_mode,
        "judge_mode": judge_mode,
        "govmem_selection_policy": govmem_selection_policy,
        **({"model": _llm_model_name()} if answer_mode == "llm" or judge_mode == "llm" else {}),
        "total_examples": total_examples,
        "baseline_evidence_hit_count": baseline_evidence_hit_count,
        "govmem_evidence_hit_count": govmem_evidence_hit_count,
        "govmem_answer_evidence_available_count": answer_evidence_available_count,
        "govmem_not_selected_answer_evidence_count": not_selected_answer_evidence_count,
        "govmem_answer_evidence_available_but_not_selected_count": answer_evidence_available_but_not_selected_count,
        "govmem_oracle_selection_used_count": oracle_selection_used_count,
        "govmem_filtered_answer_evidence_count": filtered_answer_evidence_count,
        "govmem_not_yet_valid_count": not_yet_valid_count,
        "govmem_not_yet_valid_answer_evidence_count": not_yet_valid_answer_evidence_count,
        "baseline_correct_count": baseline_correct_count,
        "govmem_correct_count": govmem_correct_count,
        "both_correct_count": both_correct_count,
        "baseline_only_correct_count": baseline_only_correct_count,
        "govmem_only_correct_count": govmem_only_correct_count,
        "both_wrong_count": both_wrong_count,
        "baseline_evidence_hit_rate": _rate(baseline_evidence_hit_count),
        "govmem_evidence_hit_rate": _rate(govmem_evidence_hit_count),
        "delta_evidence_hit_rate": _rate(govmem_evidence_hit_count - baseline_evidence_hit_count),
        "govmem_filtered_answer_evidence_rate": _rate(filtered_answer_evidence_count),
        "baseline_underspecified_count": baseline_underspecified_count,
        "govmem_underspecified_count": govmem_underspecified_count,
        "llm_answer_underspecified_count": llm_answer_underspecified_count,
        "baseline_accuracy": _rate(baseline_correct_count),
        "govmem_accuracy": _rate(govmem_correct_count),
        "delta_accuracy": _rate(govmem_correct_count - baseline_correct_count),
        "average_baseline_retrieved_memories": average_baseline_retrieved_memories,
        "average_govmem_used_memories": average_govmem_used_memories,
        "average_govmem_filtered_memories": average_govmem_filtered_memories,
        "average_memory_jaccard": average_memory_jaccard,
        "identical_memory_set_count": identical_memory_set_count,
        "identical_memory_set_rate": _rate(identical_memory_set_count),
        "average_baseline_only_memory_count": average_baseline_only_memory_count,
        "average_govmem_only_memory_count": average_govmem_only_memory_count,
        "average_not_selected_answer_evidence_memories": average_not_selected_answer_evidence_memories,
        "average_selected_non_answer_memories_when_answer_evidence_available": average_selected_non_answer_memories_when_answer_evidence_available,
        **(
            {
                "direct_gold_substring_count": direct_gold_substring_count,
                "first_between_options_count": first_between_options_count,
                "count_before_event_count": count_before_event_count,
                "enhanced_unknown_count": enhanced_unknown_count,
            }
            if answer_mode == "enhanced_heuristic"
            else {}
        ),
    }


def _select_representative_compare_examples(compare_rows: list[dict], limit: int = 8) -> list[dict]:
    ranked = sorted(
        compare_rows,
        key=lambda row: (
            int(row["baseline_correct"] != row["govmem_correct"]) * 100
            + int(row["baseline_evidence_hit"] != row["govmem_evidence_hit"]) * 50
            + int(row["baseline_answer"] != row["govmem_answer"]) * 10
            + row["govmem_filtered_count"]
            + abs(row["baseline_retrieved_count"] - row["govmem_used_count"]),
            row["question_id"],
        ),
        reverse=True,
    )
    selected = ranked[: max(5, min(limit, len(ranked)))]
    return selected


def _dedupe_compare_rows(compare_rows: list[dict]) -> list[dict]:
    deduped: OrderedDict[str, dict] = OrderedDict()
    for row in compare_rows:
        question_id = row["question_id"]
        if question_id not in deduped:
            deduped[question_id] = row
    return list(deduped.values())


def _load_existing_compare_rows(compare_results_path: Path) -> list[dict]:
    if not compare_results_path.exists():
        return []
    with compare_results_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return _dedupe_compare_rows(payload.get("examples", []))


def _merge_compare_rows(existing_rows: list[dict], new_rows: list[dict]) -> list[dict]:
    merged: OrderedDict[str, dict] = OrderedDict()
    for row in existing_rows:
        merged[row["question_id"]] = row
    for row in new_rows:
        merged[row["question_id"]] = row
    return list(merged.values())


def _memory_text_value(memory: dict) -> str:
    text = memory.get("text", memory.get("content", ""))
    return "" if text is None else str(text)


def _memory_id_list(memories: list[dict]) -> list[str]:
    return [str(memory.get("memory_id")) for memory in memories if memory.get("memory_id") is not None]


def _question_keywords(question: str) -> set[str]:
    keywords: set[str] = set()
    raw_question = question or ""
    for match in re.findall(r'"([^"]+)"', raw_question):
        normalized = normalize_text(match)
        if normalized:
            keywords.add(normalized)
    for match in re.findall(r"'([^']+)'", raw_question):
        normalized = normalize_text(match)
        if normalized:
            keywords.add(normalized)
    for match in re.findall(r"\b(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)+)\b", raw_question):
        normalized = normalize_text(match)
        if normalized:
            keywords.add(normalized)
    if _parse_two_item_choice(raw_question):
        left, right = _parse_two_item_choice(raw_question)
        if left:
            keywords.add(normalize_text(left))
        if right:
            keywords.add(normalize_text(right))
    stopwords = {
        "a", "an", "and", "are", "before", "did", "do", "does", "first", "how", "i", "in", "is", "it", "many",
        "most", "my", "or", "the", "to", "was", "what", "when", "which", "who", "why", "with", "you", "day", "days",
        "recently", "after",
    }
    for token in normalize_text(raw_question).split():
        if len(token) > 2 and token not in stopwords:
            keywords.add(token)
    return keywords


def _question_requires_temporal_focus(question: str) -> bool:
    lowered = normalize_text(question)
    return any(
        phrase in lowered
        for phrase in [
            "first",
            "before",
            "after",
            "most recently",
            "how many days",
            "how many",
            "when",
        ]
    )


_ACTION_CANONICALS = {
    "buy",
    "clean",
    "complete",
    "count",
    "create",
    "deliver",
    "eat",
    "attend",
    "finish",
    "fix",
    "play",
    "participate",
    "repair",
    "spend",
    "start",
    "setup",
    "visit",
}

_ACTION_ALIASES = {
    "bought": "buy",
    "buy": "buy",
    "boughten": "buy",
    "purchase": "buy",
    "purchased": "buy",
    "purchasing": "buy",
    "spent": "spend",
    "spending": "spend",
    "cleaned": "clean",
    "cleaning": "clean",
    "finished": "finish",
    "finishing": "finish",
    "attended": "attend",
    "attending": "attend",
    "visited": "visit",
    "visiting": "visit",
    "started": "start",
    "starting": "start",
    "participated": "participate",
    "participating": "participate",
    "repaired": "repair",
    "repairing": "repair",
    "completed": "complete",
    "completing": "complete",
    "setup": "setup",
    "set": "setup",
}

_QUESTION_NUMBER_HINTS = (
    "how many",
    "how much",
    "most money",
    "more did i spend",
    "days",
    "before",
    "after",
)


def _canonical_action_token(token: str) -> str | None:
    token = normalize_text(token)
    if not token:
        return None
    if token == "set up":
        return "setup"
    if token in _ACTION_ALIASES:
        return _ACTION_ALIASES[token]
    if token.endswith("ing") and len(token) > 4:
        token = token[:-3]
    elif token.endswith("ed") and len(token) > 3:
        token = token[:-2]
    elif token.endswith("es") and len(token) > 3:
        token = token[:-2]
    elif token.endswith("s") and len(token) > 3:
        token = token[:-1]
    return token if token in _ACTION_CANONICALS else None


def _extract_action_terms(text: str) -> set[str]:
    normalized = normalize_text(text)
    terms: set[str] = set()
    if "set up" in normalized or "setup" in normalized:
        terms.add("setup")
    for token in normalized.split():
        canonical = _canonical_action_token(token)
        if canonical:
            terms.add(canonical)
    return terms


def _extract_numeric_values(text: str) -> list[float]:
    values: list[float] = []
    for match in re.finditer(r"\$?(\d+(?:\.\d+)?)", text or ""):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            continue
    return values


def _extract_numeric_signals(text: str) -> set[str]:
    normalized = normalize_text(text)
    signals: set[str] = set()
    if re.search(r"\$\d", text or ""):
        signals.add("currency")
    if re.search(r"\b\d+\b", normalized):
        signals.add("number")
    if any(month in normalized for month in ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december")):
        signals.add("month")
    if any(duration in normalized for duration in ("day", "days", "week", "weeks", "month", "months", "year", "years")):
        signals.add("duration")
    if any(keyword in normalized for keyword in ("more", "most", "higher", "lower", "before", "after")):
        signals.add("comparison")
    return signals


def _memory_feature_score(question: str, record, query_context) -> tuple[float, list[str]]:
    text = _memory_text(record) if isinstance(record, dict) else getattr(record, "content", "")
    question_tokens = set(normalize_text(question).split())
    memory_tokens = set(normalize_text(text).split())
    lexical_overlap = len(question_tokens & memory_tokens)
    score = float(lexical_overlap * 10)
    reasons = ["lexical_overlap"]

    keywords = _question_keywords(question)
    keyword_overlap = len(keywords & memory_tokens)
    if keyword_overlap:
        score += keyword_overlap * 6
        reasons.append("entity_overlap")

    question_actions = _extract_action_terms(question)
    memory_actions = _extract_action_terms(text)
    action_overlap = question_actions & memory_actions
    if action_overlap:
        score += 40 + (len(action_overlap) - 1) * 6
        reasons.append("predicate_overlap")
        if keywords and memory_tokens:
            score += 8
            reasons.append("predicate_entity_alignment")

    if _question_requires_temporal_focus(question):
        temporal_text = normalize_text(text)
        temporal_markers = [
            "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
            "am", "pm", "morning", "afternoon", "evening", "night", "days", "day", "weeks", "months", "years", "yesterday", "today", "last", "first",
        ]
        temporal_hit = any(marker in temporal_text for marker in temporal_markers)
        if temporal_hit:
            score += 4
            reasons.append("temporal_hint")
        relation = GovernancePolicyEngine().temporal_relation(record, query_context)
        if relation in {"before_query", "same_time"}:
            score += 4
            reasons.append("temporal_relevance")
        elif relation == "after_query":
            score -= 3
            reasons.append("temporal_relevance")

    numeric_signals = _extract_numeric_signals(question)
    memory_numeric_signals = _extract_numeric_signals(text)
    numeric_values = _extract_numeric_values(text)
    if numeric_signals and memory_numeric_signals:
        score += 4
        reasons.append("numeric_evidence")
    if numeric_values:
        score += min(max(numeric_values) / 50.0, 8.0)
        reasons.append("numeric_evidence")
    if "most money" in normalize_text(question) or "more did i spend" in normalize_text(question):
        if re.search(r"\$\d", text or "") and (keywords & memory_tokens):
            score += 6
            reasons.append("numeric_evidence")
        if numeric_values:
            score += max(numeric_values) / 25.0
            reasons.append("numeric_evidence")
    if any(hint in normalize_text(question) for hint in _QUESTION_NUMBER_HINTS):
        if numeric_signals & memory_numeric_signals:
            score += 2
            reasons.append("numeric_evidence")

    text_tokens = normalize_text(text).split()
    if text_tokens:
        unique_ratio = len(set(text_tokens)) / len(text_tokens)
        score += unique_ratio
        reasons.append("diversity_bonus")

    return score, reasons


def _governance_rerank_candidates(question: str, query_context, records, k: int = 3) -> tuple[list, list[dict]]:
    engine = GovernancePolicyEngine()
    governed = engine.apply(records, query_context)
    scored: list[dict] = []
    for record in governed:
        base_score, reasons = _memory_feature_score(question, record, query_context)
        scored.append({"record": record, "base_score": base_score, "base_reasons": reasons})

    selected: list = []
    selected_tokens: list[set[str]] = []
    selected_reasons: dict[str, list[str]] = {}
    selected_scores: dict[str, float] = {}

    remaining = scored[:]
    while remaining and len(selected) < k:
        ranked: list[tuple[float, object, list[str]]] = []
        for item in remaining:
            record = item["record"]
            adjusted_score = item["base_score"]
            adjusted_reasons = list(item["base_reasons"])
            candidate_tokens = set(normalize_text(record.content).split())
            max_overlap = 0.0
            if selected_tokens and candidate_tokens:
                max_overlap = max(
                    (len(candidate_tokens & tokens) / len(candidate_tokens | tokens) if candidate_tokens | tokens else 0.0)
                    for tokens in selected_tokens
                )
            if max_overlap > 0.85:
                adjusted_score -= 2.5
                adjusted_reasons.append(f"diversity_penalty={max_overlap:.2f}")
            ranked.append((adjusted_score, record, adjusted_reasons))
        ranked.sort(key=lambda item: (-item[0], -item[1].confidence, -item[1].timestamp.timestamp(), item[1].memory_id))
        best_adjusted_score, record, best_adjusted_reasons = ranked[0]
        selected.append(record)
        selected_tokens.append(set(normalize_text(record.content).split()))
        selected_reasons[record.memory_id] = best_adjusted_reasons
        selected_scores[record.memory_id] = best_adjusted_score
        remaining = [item for item in remaining if item["record"].memory_id != record.memory_id]

    candidate_scores = []
    selected_ids = {record.memory_id for record in selected}
    for item in scored:
        record = item["record"]
        score = selected_scores.get(record.memory_id, item["base_score"])
        reasons = selected_reasons.get(record.memory_id, item["base_reasons"])
        candidate_scores.append(
            {
                "memory_id": record.memory_id,
                "selection_score": score,
                "score": score,
                "selected": record.memory_id in selected_ids,
                "selection_reasons": selected_reasons.get(record.memory_id, reasons),
                "reason": "; ".join(selected_reasons.get(record.memory_id, reasons)),
                "policy": "governance_rerank",
            }
        )
    return selected, candidate_scores


def _oracle_evidence_selection(question: str, gold_answer, query_context, records, k: int = 3) -> tuple[list, list[dict], bool]:
    engine = GovernancePolicyEngine()
    governed = engine.apply(records, query_context)
    answer_hits = [record for record in governed if evidence_hit(gold_answer, [record.content])]
    ranked, candidate_scores = _governance_rerank_candidates(question, query_context, governed, k=len(governed))
    promoted_ids = []
    selected = []
    for record in answer_hits:
        if record.memory_id not in promoted_ids:
            promoted_ids.append(record.memory_id)
            selected.append(record)
    for record in ranked:
        if record.memory_id in promoted_ids:
            continue
        selected.append(record)
    selected = selected[:k]
    oracle_selection_used = bool(answer_hits)
    promoted_set = set(promoted_ids)
    for item in candidate_scores:
        item["selected"] = item["memory_id"] in {record.memory_id for record in selected}
        if item["memory_id"] in promoted_set:
            item["reason"] = (item["reason"] + "; " if item["reason"] else "") + "oracle_evidence_promotion"
            item["selection_reasons"] = list(item.get("selection_reasons", [])) + ["oracle_evidence_promotion"]
            item["policy"] = "oracle_evidence"
    return selected, candidate_scores, oracle_selection_used


def _default_govmem_selection(question: str, query_context, records, k: int = 3) -> tuple[list, list[dict]]:
    selected = GovMemRetriever(GovernancePolicyEngine()).retrieve(records, question, query_context, k=k)
    candidate_scores = [
        {
            "memory_id": record.memory_id,
            "selection_score": lexical_score(question, record.content),
            "score": lexical_score(question, record.content),
            "selected": record in selected,
            "selection_reasons": ["default_top_k"],
            "reason": "default_top_k",
            "policy": "default",
        }
        for record in GovernancePolicyEngine().apply(records, query_context)
    ]
    return selected, candidate_scores


def _compare_memory_overlap_metrics(row: dict) -> dict:
    baseline_ids = list(row.get("baseline_memory_ids") or _memory_id_list(row.get("baseline_retrieved_memories", [])))
    govmem_ids = list(row.get("govmem_used_memory_ids") or _memory_id_list(row.get("govmem_used_memories", [])))
    baseline_set = set(baseline_ids)
    govmem_set = set(govmem_ids)
    overlap_ids = [memory_id for memory_id in baseline_ids if memory_id in govmem_set]
    baseline_only_ids = [memory_id for memory_id in baseline_ids if memory_id not in govmem_set]
    govmem_only_ids = [memory_id for memory_id in govmem_ids if memory_id not in baseline_set]
    overlap_count = len(overlap_ids)
    union_count = len(baseline_set | govmem_set)
    return {
        "baseline_memory_ids": baseline_ids,
        "govmem_used_memory_ids": govmem_ids,
        "memory_overlap_count": overlap_count,
        "memory_union_count": union_count,
        "memory_jaccard": overlap_count / union_count if union_count else 0.0,
        "memory_sets_identical": baseline_set == govmem_set,
        "baseline_only_memory_ids": baseline_only_ids,
        "govmem_only_memory_ids": govmem_only_ids,
    }


def _compare_evidence_selection_diagnostics(row: dict) -> dict:
    gold_answer = row.get("gold_answer")
    decisions = row.get("govmem_governance_decisions", [])
    decision_by_id = {decision.get("memory_id"): decision for decision in decisions}
    selected_ids = {memory.get("memory_id") for memory in row.get("govmem_used_memories", [])}

    not_selected_answer_evidence_memories = []
    for decision in decisions:
        if not decision.get("matched_answer_evidence") or decision.get("selected"):
            continue
        not_selected_answer_evidence_memories.append(
            {
                "memory_id": decision.get("memory_id"),
                "text": decision.get("text"),
                "filter_reason": decision.get("filter_reason"),
                "temporal_relation": decision.get("temporal_relation"),
                "selected": False,
                "matched_answer_evidence": True,
            }
        )

    selected_non_answer_memories = []
    for memory in row.get("govmem_used_memories", []):
        memory_id = memory.get("memory_id")
        matched_answer_evidence = evidence_hit(gold_answer, [memory.get("text", memory.get("content", ""))])
        if matched_answer_evidence:
            continue
        decision = decision_by_id.get(memory_id, {})
        selected_non_answer_memories.append(
            {
                "memory_id": memory_id,
                "text": _memory_text_value(memory),
                "filter_reason": decision.get("filter_reason"),
                "temporal_relation": decision.get("temporal_relation"),
                "selected": True,
                "matched_answer_evidence": False,
            }
        )

    return {
        "govmem_answer_evidence_available_but_not_selected": bool(not_selected_answer_evidence_memories),
        "govmem_not_selected_answer_evidence_memories": not_selected_answer_evidence_memories,
        "govmem_selected_non_answer_memories": selected_non_answer_memories,
        "govmem_answer_evidence_available_but_not_selected_count": 1 if not_selected_answer_evidence_memories else 0,
        "not_selected_answer_evidence_memory_count": len(not_selected_answer_evidence_memories),
        "selected_non_answer_memory_count": len(selected_non_answer_memories),
    }


def _compare_report_lines(summary: dict, dataset_path: str | Path) -> list[str]:
    lines = ["# Compare Report", ""]
    lines.append(f"- dataset path: `{dataset_path}`")
    lines.append(f"- total examples: {summary['total_examples']}")
    lines.append(f"- answer mode: `{summary['answer_mode']}`")
    lines.append(f"- judge mode: `{summary['judge_mode']}`")
    if summary.get("model"):
        lines.append(f"- model: `{summary['model']}`")
    lines.extend(
        [
            "",
            "| metric | value |",
            "| --- | ---: |",
            f"| baseline accuracy | {summary['baseline_accuracy']:.3f} |",
            f"| govmem accuracy | {summary['govmem_accuracy']:.3f} |",
            f"| delta accuracy | {summary['delta_accuracy']:.3f} |",
            f"| baseline correct count | {summary['baseline_correct_count']} |",
            f"| govmem correct count | {summary['govmem_correct_count']} |",
            f"| baseline evidence hit rate | {summary['baseline_evidence_hit_rate']:.3f} |",
            f"| govmem evidence hit rate | {summary['govmem_evidence_hit_rate']:.3f} |",
            f"| delta evidence hit rate | {summary['delta_evidence_hit_rate']:.3f} |",
            f"| both_correct_count | {summary['both_correct_count']} |",
            f"| baseline_only_correct_count | {summary['baseline_only_correct_count']} |",
            f"| govmem_only_correct_count | {summary['govmem_only_correct_count']} |",
            f"| both_wrong_count | {summary['both_wrong_count']} |",
            f"| baseline_underspecified_count | {summary['baseline_underspecified_count']} |",
            f"| govmem_underspecified_count | {summary['govmem_underspecified_count']} |",
            f"| llm_answer_underspecified_count | {summary['llm_answer_underspecified_count']} |",
            f"| govmem_answer_evidence_available_count | {summary['govmem_answer_evidence_available_count']} |",
            f"| govmem_filtered_answer_evidence_count | {summary['govmem_filtered_answer_evidence_count']} |",
            f"| govmem_not_yet_valid_answer_evidence_count | {summary['govmem_not_yet_valid_answer_evidence_count']} |",
            f"| govmem_not_selected_answer_evidence_count | {summary['govmem_not_selected_answer_evidence_count']} |",
        ]
    )
    lines.extend(
        [
            "",
            "## GovMem Selection Policy",
            "",
            f"- policy name: `{summary['govmem_selection_policy']}`",
            f"- average memory Jaccard: {summary['average_memory_jaccard']:.3f}",
            f"- identical memory set rate: {summary['identical_memory_set_rate']:.3f}",
            f"- answer evidence available but not selected count: {summary['govmem_answer_evidence_available_but_not_selected_count']}",
            f"- not-selected answer evidence count: {summary['govmem_not_selected_answer_evidence_count']}",
            f"- oracle selection used count: {summary.get('govmem_oracle_selection_used_count', 0)}",
            "",
            "## Memory Overlap",
            "",
            f"- average memory Jaccard: {summary['average_memory_jaccard']:.3f}",
            f"- identical memory set count: {summary['identical_memory_set_count']}",
            f"- identical memory set rate: {summary['identical_memory_set_rate']:.3f}",
            f"- average baseline-only memory count: {summary['average_baseline_only_memory_count']:.2f}",
            f"- average GovMem-only memory count: {summary['average_govmem_only_memory_count']:.2f}",
            "",
            "## GovMem Evidence Selection Misses",
            "",
            f"- answer evidence available but not selected count: {summary['govmem_answer_evidence_available_but_not_selected_count']}",
            f"- not-selected answer evidence count: {summary['govmem_not_selected_answer_evidence_count']}",
            f"- average not-selected answer evidence memories: {summary['average_not_selected_answer_evidence_memories']:.2f}",
            f"- average selected non-answer memories when answer evidence available: {summary['average_selected_non_answer_memories_when_answer_evidence_available']:.2f}",
            "- These cases show where answer-bearing candidates were present but did not survive into the final GovMem used set.",
        ]
    )
    return lines


def _compact_memory_lines(memories: list[dict], gold_answer: str) -> list[str]:
    lines = []
    for index, item in enumerate(memories[:3], start=1):
        marker = " [ANSWER_EVIDENCE]" if evidence_hit(gold_answer, [item["text"]]) else ""
        lines.append(f"{index}. {item['memory_id']}: {item['text']}{marker}")
    if not lines:
        lines.append("none")
    return lines


def _bucket_lines_for_row(row: dict) -> list[str]:
    lines = [f"## {row['question_id']}", ""]
    lines.append(f"- question: {row['question']}")
    lines.append(f"- gold answer: {row['gold_answer']}")
    lines.append(f"- baseline answer: {row['baseline_answer']}")
    lines.append(f"- govmem answer: {row['govmem_answer']}")
    lines.append(f"- comparison outcome: {row['comparison_outcome']}")
    lines.append(f"- answer delta note: {row['answer_delta_note']}")
    lines.append("")
    lines.append("Baseline retrieved memories:")
    lines.extend(_compact_memory_lines(row["baseline_retrieved_memories"], row["gold_answer"]))
    lines.append("")
    lines.append("GovMem used memories:")
    lines.extend(_compact_memory_lines(row["govmem_used_memories"], row["gold_answer"]))
    lines.append("")
    evidence_filtered = [
        item
        for item in row.get("govmem_filtered_memories", [])
        if evidence_hit(row["gold_answer"], [item["text"]])
    ]
    lines.append("GovMem governance diagnostics:")
    lines.append(f"- baseline evidence hit: {'yes' if row['baseline_evidence_hit'] else 'no'}")
    lines.append(f"- govmem evidence hit: {'yes' if row['govmem_evidence_hit'] else 'no'}")
    lines.append(f"- answer evidence available: {'yes' if row.get('answer_evidence_available') else 'no'}")
    lines.append(f"- answer evidence selected: {'yes' if row.get('answer_evidence_selected') else 'no'}")
    lines.append(f"- answer evidence filtered: {'yes' if row.get('answer_evidence_filtered') else 'no'}")
    lines.append(f"- govmem answer evidence available count: {sum(1 for decision in row['govmem_governance_decisions'] if decision['matched_answer_evidence'])}")
    lines.append(f"- govmem filtered answer evidence count: {len(evidence_filtered)}")
    lines.append(
        f"- govmem not yet valid answer evidence count: {sum(1 for decision in row['govmem_governance_decisions'] if decision['matched_answer_evidence'] and decision['filter_reason'] == 'not_yet_valid')}"
    )
    lines.append(f"- govmem not selected answer evidence count: {sum(1 for decision in row['govmem_governance_decisions'] if decision['matched_answer_evidence'] and not decision['selected'])}")
    if evidence_filtered:
        lines.append("- filtered answer evidence memories:")
        lines.extend(_compact_memory_lines(evidence_filtered, row["gold_answer"]))
    return lines


def _write_compare_bucket_files(compare_rows: list[dict], results_dir: Path) -> None:
    buckets = {
        "both_correct.md": [row for row in compare_rows if row["comparison_outcome"] == "both_correct"],
        "govmem_only_correct.md": [row for row in compare_rows if row["comparison_outcome"] == "govmem_only_correct"],
        "baseline_only_correct.md": [row for row in compare_rows if row["comparison_outcome"] == "baseline_only_correct"],
        "both_wrong.md": [row for row in compare_rows if row["comparison_outcome"] == "both_wrong"],
    }
    for filename, rows in buckets.items():
        lines = [f"# {filename.removesuffix('.md').replace('_', ' ').title()}", ""]
        if not rows:
            lines.append("No examples.")
        for row in rows:
            lines.extend(_bucket_lines_for_row(row))
            lines.append("")
        (results_dir / filename).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_summary_cases_csv(compare_rows: list[dict], results_dir: Path) -> None:
    csv_path = results_dir / "summary_cases.csv"
    fieldnames = [
        "question_id",
        "question",
        "gold_answer",
        "baseline_answer",
        "govmem_answer",
        "baseline_correct",
        "govmem_correct",
        "comparison_outcome",
        "answer_delta_note",
        "baseline_evidence_hit",
        "govmem_evidence_hit",
        "memory_jaccard",
        "memory_sets_identical",
        "memory_overlap_count",
        "baseline_only_memory_count",
        "govmem_only_memory_count",
        "govmem_answer_evidence_available",
        "govmem_answer_evidence_selected",
        "govmem_answer_evidence_filtered",
        "govmem_answer_evidence_available_but_not_selected",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in compare_rows:
            overlap_metrics = _compare_memory_overlap_metrics(row)
            writer.writerow(
                {
                    "question_id": row["question_id"],
                    "question": row["question"],
                    "gold_answer": row["gold_answer"],
                    "baseline_answer": row["baseline_answer"],
                    "govmem_answer": row["govmem_answer"],
                    "baseline_correct": str(bool(row["baseline_correct"])).lower(),
                    "govmem_correct": str(bool(row["govmem_correct"])).lower(),
                    "comparison_outcome": row["comparison_outcome"],
                    "answer_delta_note": row["answer_delta_note"],
                    "baseline_evidence_hit": str(bool(row["baseline_evidence_hit"])).lower(),
                    "govmem_evidence_hit": str(bool(row["govmem_evidence_hit"])).lower(),
                    "memory_jaccard": f"{overlap_metrics['memory_jaccard']:.6f}",
                    "memory_sets_identical": str(bool(overlap_metrics["memory_sets_identical"])).lower(),
                    "memory_overlap_count": overlap_metrics["memory_overlap_count"],
                    "baseline_only_memory_count": len(overlap_metrics["baseline_only_memory_ids"]),
                    "govmem_only_memory_count": len(overlap_metrics["govmem_only_memory_ids"]),
                    "govmem_answer_evidence_available": str(bool(row["answer_evidence_available"])).lower(),
                    "govmem_answer_evidence_selected": str(bool(row["answer_evidence_selected"])).lower(),
                    "govmem_answer_evidence_filtered": str(bool(row["answer_evidence_filtered"])).lower(),
                    "govmem_answer_evidence_available_but_not_selected": str(bool(row.get("govmem_answer_evidence_available_but_not_selected") or (row["answer_evidence_available"] and not row["answer_evidence_selected"]))).lower(),
                }
            )


def _write_selection_misses(compare_rows: list[dict], results_dir: Path) -> Path:
    path = results_dir / "selection_misses.md"
    rows = [
        row
        for row in compare_rows
        if row.get("govmem_answer_evidence_available_but_not_selected") or (row["answer_evidence_available"] and not row["answer_evidence_selected"])
    ]
    lines = ["# Selection Misses", ""]
    lines.append("Cases where GovMem had answer evidence available but did not select it.")
    lines.append("")
    if not rows:
        lines.append("No cases.")
        lines.append("")
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return path

    for row in rows:
        lines.append(f"## {row['question_id']}")
        lines.append(f"- question: {row['question']}")
        lines.append(f"- gold answer: {row['gold_answer']}")
        lines.append(f"- baseline answer: {row['baseline_answer']}")
        lines.append(f"- govmem answer: {row['govmem_answer']}")
        lines.append(f"- baseline correct: {'yes' if row['baseline_correct'] else 'no'}")
        lines.append(f"- govmem correct: {'yes' if row['govmem_correct'] else 'no'}")
        lines.append("- note: answer evidence available but not selected")
        lines.append("")
        lines.append("GovMem selected memories:")
        selected_lines = [
            f"{index}. {item['memory_id']}: {_memory_text_value(item)}"
            for index, item in enumerate(row.get("govmem_used_memories", [])[:3], start=1)
        ]
        lines.extend(selected_lines or ["none"])
        lines.append("")
        lines.append("GovMem not-selected answer evidence memories:")
        evidence_lines = []
        for index, item in enumerate(row.get("govmem_not_selected_answer_evidence_memories") or _compare_evidence_selection_diagnostics(row)["govmem_not_selected_answer_evidence_memories"], start=1):
            details = []
            if item.get("filter_reason"):
                details.append(f"filter_reason={item['filter_reason']}")
            if item.get("temporal_relation"):
                details.append(f"temporal_relation={item['temporal_relation']}")
            suffix = f" ({', '.join(details)})" if details else ""
            evidence_lines.append(f"{index}. {item['memory_id']}: {item['text']}{suffix}")
        lines.extend(evidence_lines or ["none"])
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _write_compare_artifacts(
    compare_rows: list[dict],
    results_dir: Path,
    dataset_mode: str | None,
    answer_mode: str,
    judge_mode: str,
    dataset_path: str | Path,
    govmem_selection_policy: str = "default",
) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    compare_rows = _dedupe_compare_rows(compare_rows)
    compare_results_path = results_dir / "compare_results.json"
    with compare_results_path.open("w", encoding="utf-8") as handle:
        json.dump({"examples": compare_rows}, handle, indent=2, sort_keys=True)

    with compare_results_path.open("r", encoding="utf-8") as handle:
        persisted = json.load(handle)
    summary = _compare_summary(persisted.get("examples", []), dataset_mode, answer_mode, judge_mode, govmem_selection_policy=govmem_selection_policy)

    summary_path = results_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    examples_path = results_dir / "examples.md"
    lines = ["# Compare Examples", ""]
    if dataset_mode == "official_longmemeval":
        lines.extend(
            [
                "This is official LongMemEval converted data. The comparison focuses on retrieval and evidence behavior; it is not a full QA benchmark evaluation.",
                f"Answer mode: {answer_mode}",
                f"Judge mode: {judge_mode}",
            ]
        )
        if "model" in summary:
            lines.append(f"Model: {summary['model']}")
        lines.append("")
    for row in _select_representative_compare_examples(compare_rows):
        lines.append(f"## {row['question_id']}")
        lines.append(f"Question: {row['question']}")
        lines.append(f"Gold answer: {row['gold_answer']}")
        baseline_marker = " [UNDERSPECIFIED]" if row.get("baseline_underspecified") else ""
        govmem_marker = " [UNDERSPECIFIED]" if row.get("govmem_underspecified") else ""
        lines.append(f"Comparison outcome: {row.get('comparison_outcome', 'unknown')}")
        lines.append(f"Answer delta note: {row.get('answer_delta_note', 'unknown')}")
        lines.append(f"Baseline answer: {row['baseline_answer']}{baseline_marker}")
        lines.append(f"GovMem answer: {row['govmem_answer']}{govmem_marker}")
        lines.append("Answer detail:")
        lines.append(f"- baseline: {row.get('baseline_answer_mode_detail', 'unknown')}")
        lines.append(f"- govmem: {row.get('govmem_answer_mode_detail', 'unknown')}")
        lines.append("Judge detail:")
        lines.append(f"- baseline: {row.get('baseline_judge_detail', 'unknown')}")
        lines.append(f"- govmem: {row.get('govmem_judge_detail', 'unknown')}")
        lines.append(f"Baseline correct: {'yes' if row['baseline_correct'] else 'no'}")
        lines.append(f"GovMem correct: {'yes' if row['govmem_correct'] else 'no'}")
        lines.append(f"Baseline evidence hit: {'yes' if row['baseline_evidence_hit'] else 'no'}")
        lines.append(f"GovMem evidence hit: {'yes' if row['govmem_evidence_hit'] else 'no'}")
        lines.append("Baseline retrieved memories:")
        lines.extend(_compact_memory_lines(row["baseline_retrieved_memories"], row["gold_answer"]))
        lines.append("GovMem used memories:")
        lines.extend(_compact_memory_lines(row["govmem_used_memories"], row["gold_answer"]))
        answer_evidence_reasons = row.get("answer_evidence_filter_reasons", [])
        lines.append("GovMem governance diagnostics:")
        lines.append(f"- answer evidence available: {'yes' if row.get('answer_evidence_available') else 'no'}")
        lines.append(f"- answer evidence selected: {'yes' if row.get('answer_evidence_selected') else 'no'}")
        lines.append(f"- answer evidence filtered: {'yes' if row.get('answer_evidence_filtered') else 'no'}")
        lines.append(f"- filter reasons for answer evidence: {', '.join(answer_evidence_reasons) if answer_evidence_reasons else 'none'}")
        lines.append(f"Why: {row['notes']}")
        lines.append("")
    examples_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    _write_compare_bucket_files(compare_rows, results_dir)
    _write_summary_cases_csv(compare_rows, results_dir)
    selection_misses_path = _write_selection_misses(compare_rows, results_dir)
    report_path = results_dir / "report.md"
    report_path.write_text("\n".join(_compare_report_lines(summary, dataset_path)).rstrip() + "\n", encoding="utf-8")

    return {
        "summary": summary,
        "summary_path": summary_path,
        "results_path": compare_results_path,
        "examples_path": examples_path,
        "report_path": report_path,
        "selection_misses_path": selection_misses_path,
    }


def _slice_examples(examples: list[DatasetExample], start_index: int = 0, max_examples: int | None = None) -> list[DatasetExample]:
    sliced = examples[start_index:]
    if max_examples is not None:
        sliced = sliced[:max_examples]
    return sliced


def run_demo(
    dataset_path: str | Path,
    results_dir: str | Path = "results",
    k: int = 3,
    answer_mode: str = "heuristic",
    start_index: int = 0,
    max_examples: int | None = None,
):
    examples = _slice_examples(load_dataset(dataset_path), start_index=start_index, max_examples=max_examples)
    dataset_mode = _infer_dataset_mode(examples)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    no_memory = NoMemoryBaseline()
    naive = NaiveMemoryBaseline()
    warehouse = WarehouseMemoryNoGovernanceBaseline()
    govmem_retriever = GovMemRetriever(GovernancePolicyEngine())
    agent = DeterministicMemoryAgent()

    methods = [
        _evaluate_method("no_memory", examples, no_memory, agent, k=k),
        _evaluate_method("naive_memory", examples, naive, agent, k=k),
        _evaluate_method("warehouse_no_governance", examples, warehouse, agent, k=k),
        _evaluate_method("govmem", examples, govmem_retriever, agent, k=k),
    ]

    results = {"methods": methods}
    results_path = results_dir / "demo_results.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)

    summary_lines = _build_summary_lines(methods, results_dir, dataset_mode)
    summary_path = results_dir / "demo_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    _write_traces_and_report(methods, results_dir)
    return results


def _evaluate_single_mode_examples(
    examples: list[DatasetExample],
    mode: str,
    k: int = 3,
    answer_mode: str = "heuristic",
    judge_mode: str = "heuristic",
) -> tuple[list[dict], dict]:
    baseline = NaiveMemoryBaseline()
    govmem_retriever = GovMemRetriever(GovernancePolicyEngine())
    llm_generator = build_answer_generator("llm") if answer_mode == "llm" else None
    judge = build_judge(judge_mode)

    rows: list[dict] = []

    for example in examples:
        context = to_query_context(example)
        metadata_by_id = {metadata["memory_id"]: metadata for metadata in example.memory_metadata}

        if mode == "baseline":
            retrieved = baseline.retrieve(example.memories, example.question, context, k=k)
            filtered_records: list = []
        elif mode == "govmem":
            retrieved = govmem_retriever.retrieve(example.memories, example.question, context, k=k)
            governed_candidates = govmem_retriever.policy_engine.apply(example.memories, context)
            governed_ids = {record.memory_id for record in governed_candidates}
            filtered_records = [record for record in example.memories if record.memory_id not in governed_ids]
        else:
            raise ValueError(f"Unsupported single mode: {mode}")

        if answer_mode == "llm":
            selected_memories = [{"text": record.content} for record in retrieved]
            prediction = llm_generator.generate(example.question, selected_memories, example.answer).answer
        else:
            prediction = _answer_from_records(answer_mode, example.answer, example.question, retrieved)
        judge_result = judge.evaluate(example.question, example.answer, prediction)
        correct = judge_result.correct
        used_memories = _serializable_records(retrieved, metadata_by_id)
        filtered_memories = _serializable_records(filtered_records, metadata_by_id)
        rows.append(
            {
                "question_id": example.id,
                "question": example.question,
                "gold_answer": example.answer,
                "prediction": prediction,
                "judge_detail": judge_result.detail,
                "used_memories": used_memories,
                "filtered_memories": filtered_memories,
                "correct": correct,
                "note": (
                    "GovMem applied governance filtering before retrieval." if mode == "govmem" and filtered_memories else
                    "GovMem behaved the same as the baseline on this example." if mode == "govmem" else
                    "Baseline retrieval without governance."
                ),
            }
        )

    total_examples = len(rows)
    correct_count = sum(1 for row in rows if row["correct"])
    average_used = sum(len(row["used_memories"]) for row in rows) / total_examples if total_examples else 0.0
    average_filtered = sum(len(row["filtered_memories"]) for row in rows) / total_examples if total_examples else 0.0
    summary = {
        "mode": mode,
        "total_examples": total_examples,
        "correct_count": correct_count,
        "accuracy": correct_count / total_examples if total_examples else 0.0,
        "average_retrieved_memories": average_used,
        "judge_mode": judge_mode,
    }
    if answer_mode == "llm" or judge_mode == "llm":
        summary["model"] = _llm_model_name()
    if mode == "govmem":
        summary["average_govmem_used_memories"] = average_used
        summary["average_govmem_filtered_memories"] = average_filtered
    return rows, summary


def _write_single_mode_outputs(mode: str, rows: list[dict], summary: dict, results_dir: Path, dataset_mode: str | None) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "summary.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    examples_path = results_dir / "examples.md"
    lines = [f"# {mode.title()} Examples", ""]
    if dataset_mode == "official_longmemeval":
        lines.extend(
            [
                "This run uses official LongMemEval converted data and should be interpreted as retrieval/evidence evaluation, not full QA evaluation.",
                "",
            ]
        )
    for row in rows[:10]:
        lines.append(f"## {row['question_id']}")
        lines.append(f"Question: {row['question']}")
        lines.append(f"Gold answer: {row['gold_answer']}")
        lines.append(f"Prediction: {row['prediction']}")
        lines.append(f"Correct: {'yes' if row['correct'] else 'no'}")
        lines.append(f"Judge detail: {row['judge_detail']}")
        lines.append(f"Used memories: {', '.join(item['memory_id'] for item in row['used_memories']) or 'none'}")
        if mode == "govmem":
            lines.append(f"Filtered memories: {', '.join(item['memory_id'] for item in row['filtered_memories']) or 'none'}")
        lines.append(f"Note: {row['note']}")
        lines.append("")
    examples_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    return {"summary": summary, "summary_path": results_path, "examples_path": examples_path, "rows": rows}


def run_mode(
    dataset_path: str | Path,
    results_dir: str | Path = "results",
    k: int = 3,
    mode: str = "demo",
    answer_mode: str = "heuristic",
    judge_mode: str = "heuristic",
    govmem_selection_policy: str = "default",
    start_index: int = 0,
    max_examples: int | None = None,
    resume: bool = False,
):
    if mode == "demo":
        return run_demo(
            dataset_path,
            results_dir=results_dir,
            k=k,
            answer_mode=answer_mode,
            start_index=start_index,
            max_examples=max_examples,
        )

    examples = _slice_examples(load_dataset(dataset_path), start_index=start_index, max_examples=max_examples)
    dataset_mode = _infer_dataset_mode(examples)
    results_dir = Path(results_dir)

    if mode == "compare":
        compare_results_path = results_dir / "compare_results.json"
        existing_rows = _load_existing_compare_rows(compare_results_path) if resume else []
        existing_ids = {row["question_id"] for row in existing_rows}
        missing_examples = [example for example in examples if example.id not in existing_ids]
        new_rows = _evaluate_compare_examples(
            missing_examples,
            k=k,
            answer_mode=answer_mode,
            judge_mode=judge_mode,
            govmem_selection_policy=govmem_selection_policy,
        )
        compare_rows = _merge_compare_rows(existing_rows, new_rows) if resume else new_rows
        outputs = _write_compare_artifacts(
            compare_rows,
            results_dir,
            dataset_mode,
            answer_mode,
            judge_mode,
            dataset_path,
            govmem_selection_policy=govmem_selection_policy,
        )
        return {"mode": mode, "summary": outputs["summary"], "rows": compare_rows, **outputs}

    if mode in {"baseline", "govmem"}:
        rows, summary = _evaluate_single_mode_examples(examples, mode, k=k, answer_mode=answer_mode, judge_mode=judge_mode)
        outputs = _write_single_mode_outputs(mode, rows, summary, results_dir, dataset_mode)
        return {"mode": mode, "summary": summary, "rows": rows, **outputs}

    raise ValueError(f"Unsupported mode: {mode}")
