from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable

from .metrics import answer_matches, exact_match, normalize_text


def _memory_text(memory) -> str:
    if isinstance(memory, str):
        return memory
    return memory.get("text", memory.get("content", ""))


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = normalize_text(text)
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return False
    pattern = r"(?<!\w)" + re.escape(normalized_phrase) + r"(?!\w)"
    return re.search(pattern, normalized_text) is not None


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    detail: str


@dataclass(frozen=True)
class JudgeResult:
    correct: bool
    detail: str
    rationale: str | None = None


class BaseAnswerGenerator:
    mode: str

    def generate(self, question: str, memories: list[dict], gold_answer: str | None = None) -> AnswerResult:
        raise NotImplementedError


class HeuristicAnswerGenerator(BaseAnswerGenerator):
    mode = "heuristic"

    def generate(self, question: str, memories: list[dict], gold_answer: str | None = None) -> AnswerResult:
        from . import runner as runner_mod

        answer = runner_mod._answer_from_memory_objects("heuristic", gold_answer or "", question, memories)
        return AnswerResult(answer=answer, detail=runner_mod._answer_from_memory_objects_with_detail("heuristic", gold_answer or "", question, memories)[1])


class OracleAnswerGenerator(BaseAnswerGenerator):
    mode = "oracle"

    def generate(self, question: str, memories: list[dict], gold_answer: str | None = None) -> AnswerResult:
        from . import runner as runner_mod

        answer, detail = runner_mod._answer_from_memory_objects_with_detail("oracle", gold_answer or "", question, memories)
        return AnswerResult(answer=answer, detail=detail)


class EnhancedHeuristicAnswerGenerator(BaseAnswerGenerator):
    mode = "enhanced_heuristic"

    def generate(self, question: str, memories: list[dict], gold_answer: str | None = None) -> AnswerResult:
        from . import runner as runner_mod

        answer, detail = runner_mod._answer_from_memory_objects_with_detail("enhanced_heuristic", gold_answer or "", question, memories)
        return AnswerResult(answer=answer, detail=detail)


class OpenAIChatClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, timeout: int = 60) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "LLM mode requires OPENAI_API_KEY. Use an offline answer/judge mode instead, or set OPENAI_API_KEY to enable LLM mode."
            )
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
        self.timeout = timeout

    def complete(self, *, model: str, messages: list[dict], temperature: float = 0.0) -> str:
        payload = json.dumps(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.base_url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("LLM response did not contain a message content field.") from exc


class LLMAnswerGenerator(BaseAnswerGenerator):
    mode = "llm"

    def __init__(self, client: OpenAIChatClient | None = None, model: str | None = None) -> None:
        self.client = client or OpenAIChatClient()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def generate(self, question: str, memories: list[dict], gold_answer: str | None = None) -> AnswerResult:
        messages = [
            {
                "role": "system",
                "content": (
                    "You answer questions using only the provided memory snippets. "
                    "Return the most specific answer span supported by the snippets. "
                    "Preserve important modifiers such as color, brand, object type, person or pet names, dates, times, quantities, titles, and event names. "
                    "For questions that ask which of two options happened first, return exactly the option that happened first. "
                    "For count questions, return only the number. "
                    "If the answer cannot be determined from the snippets, answer \"unknown\". "
                    "Do not use outside knowledge."
                ),
            },
            {
                "role": "user",
                "content": _render_answer_prompt(question, memories),
            },
        ]
        content = self.client.complete(model=self.model, messages=messages, temperature=0.0)
        answer = _postprocess_answer_text(content)
        return AnswerResult(answer=answer, detail="llm_response")


class BaseJudge:
    mode: str

    def evaluate(self, question: str, gold_answer: str, prediction: str) -> JudgeResult:
        raise NotImplementedError


class HeuristicJudge(BaseJudge):
    mode = "heuristic"

    def evaluate(self, question: str, gold_answer: str, prediction: str) -> JudgeResult:
        if exact_match(prediction, gold_answer):
            return JudgeResult(correct=True, detail="exact_match")
        if answer_matches(prediction, gold_answer):
            return JudgeResult(correct=True, detail="substring_match")
        return JudgeResult(correct=False, detail="mismatch")


class LLMJudge(BaseJudge):
    mode = "llm"

    def __init__(self, client: OpenAIChatClient | None = None, model: str | None = None) -> None:
        self.client = client or OpenAIChatClient()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._heuristic = HeuristicJudge()

    def evaluate(self, question: str, gold_answer: str, prediction: str) -> JudgeResult:
        messages = [
            {
                "role": "system",
                "content": "Given a question, gold answer, and predicted answer, decide whether the predicted answer is semantically equivalent to the gold answer. Return JSON: {\"correct\": true/false, \"rationale\": \"...\"}.",
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\nGold answer:\n{gold_answer}\n\nPredicted answer:\n{prediction}\n\nReturn only JSON."
                ),
            },
        ]
        content = self.client.complete(model=self.model, messages=messages, temperature=0.0)
        try:
            parsed = json.loads(content)
            correct = bool(parsed.get("correct"))
            rationale = str(parsed.get("rationale", "")).strip() or None
            detail = f"llm:{rationale}" if rationale else "llm"
            return JudgeResult(correct=correct, detail=detail, rationale=rationale)
        except Exception:
            fallback = self._heuristic.evaluate(question, gold_answer, prediction)
            return JudgeResult(correct=fallback.correct, detail=f"invalid_json_fallback:{fallback.detail}", rationale="invalid_json_fallback")


def _render_answer_prompt(question: str, memories: Iterable[dict]) -> str:
    lines = [
        "Question:",
        question,
        "",
        "Memory snippets:",
        "Use only these snippets.",
        "Prefer the most specific answer span supported by the snippets.",
        "Preserve important modifiers such as color, brand, object type, person or pet names, dates, times, quantities, titles, and event names.",
        "If the question asks which of two options happened first, return exactly one option.",
        "If the question asks for a count, return only the number.",
    ]
    for index, memory in enumerate(memories, start=1):
        lines.append(f"{index}. {_memory_text(memory)}")
    lines.append("")
    lines.append("Return only the answer, no explanation.")
    return "\n".join(lines)


def _postprocess_answer_text(text: str) -> str:
    answer = (text or "").strip()
    if not answer:
        return "unknown"
    if normalize_text(answer) == "unknown":
        return "unknown"
    if len(answer) >= 2 and answer[0] == answer[-1] and answer[0] in {'"', "'", "`"}:
        answer = answer[1:-1].strip()
    answer = answer.strip().rstrip(" \t\r\n.,;:!?")
    answer = re.sub(r"\s+", " ", answer).strip()
    if not answer:
        return "unknown"
    if normalize_text(answer) == "unknown":
        return "unknown"
    return answer


def build_answer_generator(answer_mode: str, client: OpenAIChatClient | None = None, model: str | None = None) -> BaseAnswerGenerator:
    if answer_mode == "heuristic":
        return HeuristicAnswerGenerator()
    if answer_mode == "oracle":
        return OracleAnswerGenerator()
    if answer_mode == "enhanced_heuristic":
        return EnhancedHeuristicAnswerGenerator()
    if answer_mode == "llm":
        return LLMAnswerGenerator(client=client, model=model)
    raise ValueError(f"Unsupported answer mode: {answer_mode}")


def build_judge(judge_mode: str, client: OpenAIChatClient | None = None, model: str | None = None) -> BaseJudge:
    if judge_mode == "heuristic":
        return HeuristicJudge()
    if judge_mode == "llm":
        return LLMJudge(client=client, model=model)
    raise ValueError(f"Unsupported judge mode: {judge_mode}")
