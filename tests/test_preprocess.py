"""Unit test untuk modul preprocessing teks & scoring sentimen."""

from src.sentiment.lexicon_scorer import InSetLexiconScorer
from src.sentiment.preprocess_text import (
    clean_text,
    normalize_slang,
    remove_stopwords,
)


def test_clean_text():
    raw = "Beli $BBCA di https://stockbit.com! @user123 baguss 👍 100%"
    cleaned = clean_text(raw)
    assert "https" not in cleaned
    assert "$bbca" not in cleaned
    assert "@user123" not in cleaned
    assert "baguss" in cleaned


def test_normalize_slang():
    raw = "cuan bgt gak rugi cl tp"
    normalized = normalize_slang(raw)
    assert "untung" in normalized
    assert "banget" in normalized
    assert "tidak" in normalized
    assert "cut loss" in normalized
    assert "take profit" in normalized


def test_remove_stopwords_preserves_negation():
    raw = "saham ini tidak bagus dan sangat jelek"
    filtered = remove_stopwords(raw)
    assert "tidak" in filtered


def test_lexicon_scorer():
    scorer = InSetLexiconScorer()
    res_pos = scorer.score_text("cuan untung terbang bagus")
    assert res_pos["label"] == 1
    assert res_pos["score"] > 0

    res_neg = scorer.score_text("rugi loss longsor jelek")
    assert res_neg["label"] == -1
    assert res_neg["score"] < 0


def test_lexicon_scorer_negation_handling():
    scorer = InSetLexiconScorer()
    res_negated_pos = scorer.score_text("tidak bagus")
    assert res_negated_pos["score"] < 0
    assert res_negated_pos["label"] == -1

    res_negated_neg = scorer.score_text("tidak rugi")
    assert res_negated_neg["score"] > 0
    assert res_negated_neg["label"] == 1
