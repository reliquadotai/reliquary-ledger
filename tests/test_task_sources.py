import reliquary_inference.dataset.task_sources as task_sources_module
from reliquary_inference.dataset.task_sources import build_task_source


def _window_context(source_id: str) -> dict:
    return {
        "window_id": 1,
        "block_hash": "00" * 32,
        "public_randomness": "11" * 32,
        "task_source": source_id,
        "model_ref": "toy://local-inference-v1",
        "dataset_name": "missing-dataset",
        "dataset_split": "train",
    }


def test_dataset_prompts_source_is_deterministic() -> None:
    source = build_task_source("dataset_prompts")
    batch_a = source.build_window_batch(_window_context("dataset_prompts"), count=3)
    batch_b = source.build_window_batch(_window_context("dataset_prompts"), count=3)
    assert [task["task_id"] for task in batch_a["tasks"]] == [task["task_id"] for task in batch_b["tasks"]]


def test_reasoning_tasks_source_is_deterministic() -> None:
    source = build_task_source("reasoning_tasks")
    batch_a = source.build_window_batch(_window_context("reasoning_tasks"), count=4)
    batch_b = source.build_window_batch(_window_context("reasoning_tasks"), count=4)
    assert [task["task_id"] for task in batch_a["tasks"]] == [task["task_id"] for task in batch_b["tasks"]]
    assert batch_a["tasks"][0]["verification_mode"] == "exact_final_answer"


def test_reasoning_tasks_source_uses_model_chat_rendering(monkeypatch) -> None:
    class StubTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
            assert tokenize is False
            assert add_generation_prompt is True
            return f"CHAT::{messages[-1]['content']}"

    monkeypatch.setattr(task_sources_module, "load_tokenizer_for_model", lambda model_ref: StubTokenizer())
    source = build_task_source("reasoning_tasks")
    window_context = _window_context("reasoning_tasks")
    window_context["model_ref"] = "hf://stub-model"
    batch = source.build_window_batch(window_context, count=1)
    assert batch["tasks"][0]["prompt"].startswith("CHAT::")
    assert "source_prompt" in batch["tasks"][0]
