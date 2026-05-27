from working_gpt.tokenizer import CharTokenizer


def test_char_tokenizer_roundtrip():
    text = "risk return"
    tokenizer = CharTokenizer.from_text(text)
    assert tokenizer.decode(tokenizer.encode(text)) == text

