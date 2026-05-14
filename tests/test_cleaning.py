from src.data.cleaning import clean_records, normalize_text


def test_normalize_text_removes_residual_html_literals():
    text = normalize_text('Hello <script>alert("x")</script> <a href="x">link</a>')

    assert "<script" not in text
    assert "<a " not in text


def test_clean_records_filters_cjk_heavy_english_texts():
    rows = [
        {
            "id": "mixed",
            "text": "English words " * 30 + "中文中文中文中文中文中文中文中文中文中文",
        }
    ]

    assert list(clean_records(rows, language_allowlist=("en",), min_words=20)) == []


def test_clean_records_filters_markup_reference_pages():
    rows = [
        {
            "id": "bbcode",
            "text": (
                "Example Usage [code] value [/code] [url] value [/url] "
                "<a href=\"testing.html\" target=\"_blank\"> Testing </a> "
                "ordinary words " * 30
            ),
        }
    ]

    assert list(clean_records(rows, language_allowlist=("en",), min_words=20)) == []
