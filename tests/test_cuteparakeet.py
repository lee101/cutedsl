from __future__ import annotations

from cuteparakeet.benchmark import collect_audio_files, compute_char_error_rate, compute_word_error_rate, normalize_text


def test_normalize_text() -> None:
    assert normalize_text(" Hello,   World!! ") == "hello world"


def test_word_error_rate() -> None:
    wer = compute_word_error_rate("the quick brown fox", "the quick fox")
    assert wer == 0.25


def test_char_error_rate() -> None:
    cer = compute_char_error_rate("abc", "adc")
    assert cer == 1 / 3


def test_collect_audio_files_dedupes_and_filters(tmp_path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    keep = audio_dir / "sample.wav"
    keep.write_bytes(b"wav")
    ignore = audio_dir / "notes.txt"
    ignore.write_text("ignore", encoding="utf-8")
    explicit = audio_dir / "other.ogg"
    explicit.write_bytes(b"ogg")

    files = collect_audio_files([str(explicit), str(explicit)], str(audio_dir))

    assert files == [explicit.resolve(), keep.resolve()]
