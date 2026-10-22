from app.preprocess import normalize_text, preprocess_messages


def test_taglish_and_emoji_normalization():
    text = "Pls stop na 😡 malandi ka"
    normalized = normalize_text(text)
    assert "please" in normalized
    assert "angry" in normalized
    assert "sexually suggestive" in normalized


def test_gay_lingo_expansion():
    processed = preprocess_messages([{"speaker": "User A", "text": "Charot gurl wiz"}])
    assert "just kidding" in processed[0]["normalized"]
    assert "girl" in processed[0]["normalized"]


def test_mapping_replacements_are_not_applied_recursively():
    assert normalize_text("pakyu") == "fuck you"


def test_sexual_slang_is_normalized_without_removing_original_message():
    processed = preprocess_messages(
        [{"speaker": "User A", "text": "Send noods, chupa, seggs"}]
    )

    assert "send nude photos" in processed[0]["normalized"]
    assert "oral sex" in processed[0]["normalized"]
    assert "sex" in processed[0]["normalized"]
    assert processed[0]["original"] == "Send noods, chupa, seggs"


def test_ambiguous_skull_emoji_is_not_normalized_as_literal_death():
    assert normalize_text("💀") == "ambiguous reaction"
