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
