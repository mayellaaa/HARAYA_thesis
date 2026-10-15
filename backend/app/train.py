from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .preprocess import conversation_document, normalize_text, preprocess_messages
from .scoring import load_rules

REQUIRED_COLUMNS = {"conversation_id"}


@dataclass(frozen=True)
class Conversation:
    conversation_id: str
    text: str
    label: str


def _canonical_label(value: str, labels: set[str], line_number: int) -> str:
    normalized = re.sub(r"\s*/\s*", "/", value.strip()).casefold()
    for label in labels:
        if re.sub(r"\s*/\s*", "/", label).casefold() == normalized:
            return label
    raise ValueError(
        f"Dataset row {line_number} has unknown label {value!r}. "
        f"Use one of: {', '.join(sorted(labels))}."
    )


def load_dataset(path: Path) -> list[Conversation]:
    labels = set(load_rules()["categories"])
    conversations: dict[str, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as dataset_file:
        reader = csv.DictReader(dataset_file)
        columns = set(reader.fieldnames or [])
        text_column = "text" if "text" in columns else "conversation"
        label_column = "label" if "label" in columns else "final_label"
        missing = REQUIRED_COLUMNS - columns
        if text_column not in columns:
            missing.add("text (or conversation)")
        if label_column not in columns:
            missing.add("label (or final_label)")
        if missing:
            raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

        has_speaker = "speaker" in columns
        for line_number, row in enumerate(reader, start=2):
            conversation_id = (row.get("conversation_id") or "").strip()
            text = (row.get(text_column) or "").strip()
            original_label = (row.get(label_column) or "").strip()
            if not conversation_id or not text or not original_label:
                raise ValueError(f"Dataset row {line_number} has an empty required value.")
            label = _canonical_label(original_label, labels, line_number)

            record = conversations.setdefault(
                conversation_id,
                {
                    "label": label,
                    "messages": [],
                    "order": 0,
                    "data_origin": row.get("data_origin", ""),
                    "validation_status": row.get("validation_status", ""),
                },
            )
            if record["label"] != label:
                raise ValueError(
                    f"Conversation {conversation_id!r} has more than one label."
                )
            if "data_origin" in columns and record["data_origin"] != row.get("data_origin", ""):
                raise ValueError(
                    f"Conversation {conversation_id!r} has inconsistent data_origin values."
                )
            if "validation_status" in columns and record["validation_status"] != row.get(
                "validation_status", ""
            ):
                raise ValueError(
                    f"Conversation {conversation_id!r} has inconsistent validation_status values."
                )

            if has_speaker:
                speaker = (row.get("speaker") or "").strip()
                if speaker not in {"User A", "User B"}:
                    raise ValueError(
                        f"Dataset row {line_number} has invalid speaker {speaker!r}."
                    )
                if "order" in columns and (row.get("order") or "").strip():
                    try:
                        message_order = int(row["order"])
                    except ValueError as exc:
                        raise ValueError(
                            f"Dataset row {line_number} has a non-integer order."
                        ) from exc
                    record["messages"].append((message_order, line_number, speaker, text))
                else:
                    record["messages"].append(
                        (record["order"], line_number, speaker, text)
                    )
                    record["order"] += 1
            else:
                if conversation_id in conversations and record["messages"]:
                    raise ValueError(
                        f"Conversation-level dataset has duplicate ID {conversation_id!r}; "
                        "use speaker and optional order columns for message-level rows."
                    )
                record["messages"].append((0, line_number, "", text))

    result = []
    for conversation_id, record in conversations.items():
        ordered_messages = sorted(record["messages"], key=lambda item: (item[0], item[1]))
        if has_speaker:
            messages = [
                {"speaker": speaker, "text": text}
                for _, _, speaker, text in ordered_messages
            ]
            document = conversation_document(preprocess_messages(messages))
        else:
            document = normalize_text(ordered_messages[0][3])
        result.append(
            Conversation(
                conversation_id=conversation_id,
                text=document,
                label=record["label"],
            )
        )
    if not result:
        raise ValueError("Dataset contains no conversations.")
    return result


def dataset_profile(path: Path) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as dataset_file:
        reader = csv.DictReader(dataset_file)
        rows = list(reader)
        for column in ("data_origin", "source_type", "validation_status"):
            if column in (reader.fieldnames or []):
                profile[column] = dict(
                    sorted(
                        Counter(
                            (row.get(column) or "").strip() or "(blank)"
                            for row in rows
                        ).items()
                    )
                )
    return profile


def stratified_conversation_split(
    conversations: list[Conversation], test_fraction: float, seed: int
) -> tuple[list[Conversation], list[Conversation]]:
    if not 0 < test_fraction < 0.5:
        raise ValueError("Test fraction must be greater than 0 and less than 0.5.")

    by_label: dict[str, list[Conversation]] = defaultdict(list)
    for conversation in conversations:
        by_label[conversation.label].append(conversation)

    randomizer = random.Random(seed)
    training: list[Conversation] = []
    testing: list[Conversation] = []
    for label, group in sorted(by_label.items()):
        if len(group) < 2:
            raise ValueError(
                f"Label {label!r} needs at least two conversations for a "
                "stratified train/test split."
            )
        randomizer.shuffle(group)
        test_count = min(len(group) - 1, max(1, round(len(group) * test_fraction)))
        testing.extend(group[:test_count])
        training.extend(group[test_count:])

    randomizer.shuffle(training)
    randomizer.shuffle(testing)
    return training, testing


class ConversationDataset(Dataset):
    def __init__(self, conversations: list[Conversation], label_to_id: dict[str, int]):
        self.conversations = conversations
        self.label_to_id = label_to_id

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, index: int) -> dict[str, Any]:
        conversation = self.conversations[index]
        return {
            "text": conversation.text,
            "labels": self.label_to_id[conversation.label],
        }


def _metrics(
    expected: list[int], predicted: list[int], labels: list[str]
) -> dict[str, Any]:
    per_label: dict[str, dict[str, float | int]] = {}
    f1_values = []
    for label_id, label in enumerate(labels):
        true_positive = sum(
            actual == label_id and guess == label_id
            for actual, guess in zip(expected, predicted)
        )
        false_positive = sum(
            actual != label_id and guess == label_id
            for actual, guess in zip(expected, predicted)
        )
        false_negative = sum(
            actual == label_id and guess != label_id
            for actual, guess in zip(expected, predicted)
        )
        support = sum(actual == label_id for actual in expected)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        if support:
            f1_values.append(f1)

    accuracy = (
        sum(actual == guess for actual, guess in zip(expected, predicted)) / len(expected)
        if expected
        else 0.0
    )
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(sum(f1_values) / len(f1_values), 4) if f1_values else 0.0,
        "per_label": per_label,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    conversations = load_dataset(args.dataset)
    labels = sorted({item.label for item in conversations})
    training, testing = stratified_conversation_split(
        conversations, args.test_fraction, args.seed
    )
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = tokenizer(
            [item["text"] for item in batch],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([item["labels"] for item in batch])
        return encoded

    train_loader = DataLoader(
        ConversationDataset(training, label_to_id),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        ConversationDataset(testing, label_to_id),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            total_loss += output.loss.item()
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"training_loss={total_loss / max(len(train_loader), 1):.4f}"
        )

    model.eval()
    expected: list[int] = []
    predicted: list[int] = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels_batch = batch.pop("labels")
            logits = model(**batch).logits
            expected.extend(labels_batch.cpu().tolist())
            predicted.extend(logits.argmax(dim=1).cpu().tolist())

    metrics = _metrics(expected, predicted, labels)
    report = {
        "model_id": args.model_id,
        "seed": args.seed,
        "training_conversations": len(training),
        "test_conversations": len(testing),
        "labels": labels,
        "dataset_profile": dataset_profile(args.dataset),
        **metrics,
        "note": (
            "Held-out test metrics are for the trained contextual classifier. "
            "HARAYA's rule-based legal scoring engine remains the final decision-maker."
        ),
    }
    profile = report["dataset_profile"]
    origins = {
        *profile.get("data_origin", {}),
        *profile.get("source_type", {}),
    }
    statuses = set(profile.get("validation_status", {}))
    if any("synthetic" in origin.casefold() for origin in origins) or any(
        "review" in status.casefold() for status in statuses
    ):
        report["dataset_quality_warning"] = (
            "This dataset is marked as synthetic and/or requires expert review. "
            "Metrics are an implementation check, not evidence of real-world accuracy; "
            "validate on independently collected and expert-reviewed conversations."
        )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved classifier and evaluation report to {output_dir}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune and evaluate HARAYA's contextual mBERT classifier."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Labeled conversation CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("models/haraya-classifier"))
    parser.add_argument("--model-id", default="google-bert/bert-base-multilingual-cased")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 1:
        parser.error("epochs, batch-size, and max-length must be positive integers.")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive.")
    try:
        train(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
