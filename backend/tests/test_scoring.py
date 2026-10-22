from app.indicators import detect_indicators
from app.preprocess import conversation_document, preprocess_messages
from app.scoring import final_prediction, score_categories


def _detected(*indicators: str) -> dict[str, dict[str, bool]]:
    return {indicator: {"present": True} for indicator in indicators}


def _run(messages: list[dict]):
    processed = preprocess_messages(messages)
    detected = detect_indicators(processed, conversation_document(processed))
    scores = score_categories(detected)
    prediction, winner = final_prediction(scores)
    return prediction, winner, detected, scores


def test_cyberstalking_conversation_level():
    messages = [
        {"speaker": "User B", "text": "Please stop messaging me."},
        {"speaker": "User A", "text": "No."},
        {"speaker": "User B", "text": "I already blocked you."},
        {"speaker": "User A", "text": "I'll make another account."},
        {"speaker": "User A", "text": "I monitor your posts para malaman kung nasaan ka."},
    ]
    prediction, winner, detected, _ = _run(messages)
    assert detected["continued_after_refusal"]["present"] is True
    assert detected["multiple_accounts"]["present"] is True
    assert detected["repeated_communication"]["present"] is True
    assert prediction in {"Cyberstalking", "Incessant Messaging"}
    assert winner is not None
    assert winner["score"] >= 4


def test_shared_indicators_do_not_score_categories_without_core_evidence():
    scores = score_categories(
        _detected("electronic_communication", "gender_based_targeting", "intimidation")
    )

    assert scores
    assert all(item["score"] == 0 for item in scores)
    assert all(item["decision"] == "Insufficient evidence" for item in scores)


def test_category_specific_evidence_prevents_unrelated_category_scores():
    scores = score_categories(
        _detected("misogynistic_language", "gender_based_targeting", "intimidation")
    )
    scores_by_label = {item["label"]: item for item in scores}

    assert scores_by_label["Misogynistic Remarks/Comments"]["score"] == 9
    assert scores_by_label["Misogynistic Remarks/Comments"]["decision"] == "Classified"
    assert scores_by_label["Sexist Remarks/Comments"]["score"] == 0
    assert scores_by_label["Unwanted Sexual Remarks/Comments"]["score"] == 0


def test_incessant_messaging_does_not_also_classify_as_cyberstalking_without_stalking_evidence():
    scores = score_categories(
        _detected("repeated_communication", "continued_after_refusal")
    )
    scores_by_label = {item["label"]: item for item in scores}

    assert scores_by_label["Incessant Messaging"]["decision"] == "Classified"
    assert scores_by_label["Cyberstalking"]["score"] == 0


def test_non_harassment_greeting():
    messages = [
        {"speaker": "User A", "text": "Hi, are you free later for the group project?"},
        {"speaker": "User B", "text": "Yes, after class is fine."},
        {"speaker": "User A", "text": "Thanks, I will send the slides."},
    ]
    prediction, winner, detected, _ = _run(messages)
    assert detected["physical_harm"]["present"] is False
    assert detected["sexual_context"]["present"] is False
    assert prediction == "Non-Harassment" or (winner and winner["decision"] != "Classified")
