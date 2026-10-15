from app.indicators import detect_indicators
from app.preprocess import conversation_document, preprocess_messages
from app.scoring import final_prediction, score_categories


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
