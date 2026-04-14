from reliquary_inference.dataset.reasoning import evaluate_reasoning_trace


def _task(answer: str = "16") -> dict:
    return {"reference_answer": answer}


def test_evaluate_reasoning_trace_accepts_inline_labeled_final_answer() -> None:
    text = "Reasoning: the smaller value is 16.\nThe final answer is `Final Answer: 16`."
    result = evaluate_reasoning_trace(_task("16"), text)
    assert result["accepted"] is True
    assert result["format_ok"] is True
    assert result["final_answer"] == "16"


def test_evaluate_reasoning_trace_accepts_natural_language_final_answer_phrase() -> None:
    text = "Reasoning: 13 plus 7 is 20, then 20 times 8 is 160. The final answer is 160."
    result = evaluate_reasoning_trace(_task("160"), text)
    assert result["accepted"] is True
    assert result["format_ok"] is True
    assert result["final_answer"] == "160"


def test_evaluate_reasoning_trace_rejects_wrong_explicit_answer() -> None:
    text = "Reasoning: computed it.\nFinal Answer: 300"
    result = evaluate_reasoning_trace(_task("304"), text)
    assert result["accepted"] is False
    assert result["format_ok"] is True
    assert result["correctness_or_judge"] == 0.0
