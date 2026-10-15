import csv

import pytest

from app.train import load_dataset, stratified_conversation_split


def _write_dataset(path, labels_by_conversation):
    with path.open("w", encoding="utf-8", newline="") as dataset_file:
        writer = csv.DictWriter(
            dataset_file,
            fieldnames=["conversation_id", "speaker", "text", "label", "order"],
        )
        writer.writeheader()
        for conversation_id, label in labels_by_conversation:
            writer.writerow(
                {
                    "conversation_id": conversation_id,
                    "speaker": "User A",
                    "text": f"Message for {conversation_id}",
                    "label": label,
                    "order": 1,
                }
            )


def test_load_dataset_groups_and_orders_messages(tmp_path):
    path = tmp_path / "dataset.csv"
    with path.open("w", encoding="utf-8", newline="") as dataset_file:
        writer = csv.DictWriter(
            dataset_file,
            fieldnames=["conversation_id", "speaker", "text", "label", "order"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "conversation_id": "chat-1",
                    "speaker": "User B",
                    "text": "Please stop.",
                    "label": "Cyberstalking",
                    "order": 2,
                },
                {
                    "conversation_id": "chat-1",
                    "speaker": "User A",
                    "text": "I will keep contacting you.",
                    "label": "Cyberstalking",
                    "order": 1,
                },
            ]
        )

    conversations = load_dataset(path)

    assert len(conversations) == 1
    assert conversations[0].label == "Cyberstalking"
    assert conversations[0].text.index("User A") < conversations[0].text.index("User B")


def test_load_dataset_rejects_conflicting_conversation_labels(tmp_path):
    path = tmp_path / "dataset.csv"
    _write_dataset(path, [("chat-1", "Cyberstalking"), ("chat-1", "Non-Harassment")])

    with pytest.raises(ValueError, match="more than one label"):
        load_dataset(path)


def test_stratified_split_keeps_each_conversation_in_one_partition():
    from app.train import Conversation

    conversations = [
        Conversation(f"chat-{label}-{index}", "text", label)
        for label in ("Cyberstalking", "Non-Harassment")
        for index in range(5)
    ]

    training, testing = stratified_conversation_split(conversations, 0.2, 42)

    assert {item.label for item in training} == {"Cyberstalking", "Non-Harassment"}
    assert {item.label for item in testing} == {"Cyberstalking", "Non-Harassment"}
    assert not ({item.conversation_id for item in training} & {item.conversation_id for item in testing})


def test_stratified_split_requires_two_conversations_per_label():
    from app.train import Conversation

    with pytest.raises(ValueError, match="at least two conversations"):
        stratified_conversation_split(
            [Conversation("only-one", "text", "Cyberstalking")], 0.2, 42
        )
