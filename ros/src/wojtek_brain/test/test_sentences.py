"""speakable/split_sentences (parity with #131) and the stream assembler."""

from wojtek_brain.sentences import SentenceAssembler, speakable, split_sentences


class TestSpeakable:
    def test_strips_emoji_and_markdown(self):
        assert speakable("**Hau!** 🐕 Idę!") == "Hau! Idę!"

    def test_collapses_whitespace(self):
        assert speakable("a\n\n  b") == "a b"


class TestSplitSentences:
    def test_splits_on_sentence_ends(self):
        chunks = split_sentences(
            "To jest pierwsze pełne zdanie. A to jest drugie pełne zdanie!"
        )
        assert len(chunks) == 2

    def test_abbreviation_does_not_split(self):
        chunks = split_sentences("Widzę np. krzesło i stół w dużym pokoju obok.")
        assert len(chunks) == 1

    def test_short_first_clause_merges_forward(self):
        chunks = split_sentences("Tak. Naprawdę bardzo chętnie pójdę tam z tobą.")
        assert len(chunks) == 1


class TestSentenceAssembler:
    def feed_text(self, assembler, text, chunk=3):
        out = []
        for i in range(0, len(text), chunk):
            out.extend(assembler.feed(text[i : i + chunk]))
        return out

    def test_emits_sentence_when_next_begins(self):
        a = SentenceAssembler(min_chars=10)
        done = self.feed_text(
            a, "Pierwsze zdanie jest tutaj. Drugie zdanie wciąż trwa"
        )
        assert done == ["Pierwsze zdanie jest tutaj."]
        assert a.flush() == ["Drugie zdanie wciąż trwa"]

    def test_no_emit_before_terminator(self):
        a = SentenceAssembler()
        assert self.feed_text(a, "To zdanie nie ma jeszcze końca ani kropki") == []

    def test_abbreviation_held_back(self):
        a = SentenceAssembler(min_chars=10)
        done = self.feed_text(a, "Widzę np. krzesło przy oknie. Idę tam teraz")
        assert done == ["Widzę np. krzesło przy oknie."]

    def test_flush_empties_buffer(self):
        a = SentenceAssembler()
        a.feed("Krótki tekst")
        assert a.flush() == ["Krótki tekst"]
        assert a.flush() == []

    def test_multi_sentence_burst(self):
        a = SentenceAssembler(min_chars=10)
        done = a.feed(
            "Pierwsze pełne zdanie tutaj. Drugie pełne zdanie tutaj! A trzecie"
        )
        assert done == ["Pierwsze pełne zdanie tutaj.", "Drugie pełne zdanie tutaj!"]
