import json
from pathlib import Path

from src.conversation_store import ConversationStore, conversation_messages_from_turns


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_store_reads_legacy_json_and_ignores_zone_identifier(tmp_path: Path) -> None:
    legacy_path = tmp_path / "qa_conversation_20260516_165042.json"
    write_json(
        legacy_path,
        {
            "saved_at": "2026-05-16T16:50:42",
            "turns": [
                {
                    "created_at": "2026-05-16T16:46:26",
                    "question": "中际旭创和新易盛在光模块业务上的差异是什么？",
                    "answer": "两家公司都涉及光模块业务。",
                    "result": {"contextual_question": "中际旭创和新易盛在光模块业务上的差异是什么？"},
                }
            ],
        },
    )
    (tmp_path / "qa_conversation_20260516_165042.json:Zone.Identifier").write_text(
        "[ZoneTransfer]\n",
        encoding="utf-8",
    )

    store = ConversationStore(tmp_path)
    summaries = store.list()
    conversation = store.get("20260516_165042")

    assert len(summaries) == 1
    assert summaries[0]["id"] == "20260516_165042"
    assert summaries[0]["turn_count"] == 1
    assert conversation["title"].startswith("中际旭创和新易盛")


def test_store_creates_appends_renames_exports_and_deletes(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    conversation = store.create()
    conversation_id = conversation["id"]

    updated = store.append_turn(
        conversation_id,
        {
            "created_at": "2026-05-16T17:00:00",
            "question": "液冷产业链有哪些上市公司？",
            "answer": "英维克等公司有相关业务。",
            "result": {"contextual_question": "液冷产业链有哪些上市公司？"},
        },
    )
    renamed = store.update_title(conversation_id, "液冷产业链问答")
    markdown, filename, media_type = store.export(conversation_id, "md")

    assert updated["title"].startswith("液冷产业链")
    assert renamed["title"] == "液冷产业链问答"
    assert "液冷产业链有哪些上市公司" in markdown
    assert filename == "液冷产业链问答.md"
    assert media_type.startswith("text/markdown")

    store.delete(conversation_id)
    assert store.list() == []


def test_conversation_messages_from_turns_keeps_user_assistant_order() -> None:
    turns = [
        {"question": "第一问", "answer": "第一答"},
        {"question": "第二问", "answer": "第二答"},
    ]

    assert conversation_messages_from_turns(turns) == [
        {"role": "user", "content": "第一问"},
        {"role": "assistant", "content": "第一答"},
        {"role": "user", "content": "第二问"},
        {"role": "assistant", "content": "第二答"},
    ]
