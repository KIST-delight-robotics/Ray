"""Sentence boundary detection for streaming LLM text."""

from __future__ import annotations


class SentenceDetector:
    """Accumulates streaming text and yields complete sentences.

    Designed for English text. Detects sentence boundaries at ``.``, ``!``,
    ``?`` followed by whitespace. Skips common abbreviations and single-
    letter initials (e.g. ``Mr.``, ``U.S.``).

    A sentence is only yielded when the accumulated word count from the
    start of the buffer meets *min_flush_words*. Shorter fragments are
    held until a later boundary satisfies the threshold (or :meth:`flush`
    is called).
    """

    _ABBREVIATIONS: frozenset[str] = frozenset(
        {
            "mr",
            "mrs",
            "ms",
            "dr",
            "jr",
            "sr",
            "st",
            "vs",
            "etc",
            "prof",
            "gen",
            "sgt",
            "col",
            "lt",
            "capt",
            "maj",
            "rev",
            "hon",
            "govt",
            "dept",
            "inc",
            "corp",
            "ltd",
            "approx",
            "avg",
            "vol",
            "no",
            "fig",
        }
    )

    _SENTENCE_ENDERS: frozenset[str] = frozenset(".!?")

    def __init__(self, min_flush_words: int = 4) -> None:
        self._buffer: str = ""
        self._min_flush_words = min_flush_words

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, chunk: str) -> list[str]:
        """Append *chunk* and return any complete sentences.

        Returns an empty list when no sentence boundary has been detected
        yet or the accumulated word count is below *min_flush_words*.
        """
        self._buffer += chunk
        results: list[str] = []

        while True:
            boundary = self._find_next_boundary()
            if boundary is None:
                break

            candidate = self._buffer[:boundary]
            if self._word_count(candidate) >= self._min_flush_words:
                results.append(candidate.strip())
                self._buffer = self._buffer[boundary:].lstrip()
            else:
                # Not enough words yet — try the next boundary further out.
                next_b = self._find_next_boundary(start=boundary)
                if next_b is not None:
                    # There is a later boundary; loop will re-evaluate
                    # from the top with the same buffer.  We need to skip
                    # past the current boundary, so search from *boundary*.
                    # To avoid infinite loop, we must actually try to
                    # split at the later boundary.
                    candidate2 = self._buffer[:next_b]
                    if self._word_count(candidate2) >= self._min_flush_words:
                        results.append(candidate2.strip())
                        self._buffer = self._buffer[next_b:].lstrip()
                        continue
                    # Still not enough — keep scanning.
                    # Advance start to find even later boundaries.
                    pos = next_b
                    while True:
                        later = self._find_next_boundary(start=pos)
                        if later is None:
                            break
                        candidate_n = self._buffer[:later]
                        if self._word_count(candidate_n) >= self._min_flush_words:
                            results.append(candidate_n.strip())
                            self._buffer = self._buffer[later:].lstrip()
                            break
                        pos = later
                break

        return results

    def flush(self) -> str | None:
        """Return remaining buffer contents (end-of-stream).

        Returns ``None`` if the buffer is empty or whitespace-only.
        Resets the internal buffer.
        """
        text = self._buffer.strip()
        self._buffer = ""
        return text or None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _word_count(text: str) -> int:
        """Count whitespace-delimited words in *text*."""
        return len(text.split())

    def _find_next_boundary(self, start: int = 0) -> int | None:
        """Find position *after* the next sentence boundary in buffer.

        Scans from *start*. A boundary is a sentence-ending punctuation
        mark (``.``, ``!``, ``?``) followed by at least one whitespace
        character. Returns the index of the first whitespace character
        after the punctuation, or ``None`` if no boundary is found.
        """
        i = start
        buf = self._buffer
        length = len(buf)

        while i < length:
            ch = buf[i]
            if ch in self._SENTENCE_ENDERS:
                # Skip consecutive sentence-enders / closing quotes.
                j = i + 1
                while j < length and buf[j] in ".!?\"')\u201d\u2019":
                    j += 1

                # Boundary requires trailing whitespace.
                if j >= length:
                    # Punctuation at end of buffer — can't confirm yet.
                    return None

                if buf[j] == " " or buf[j] == "\n":
                    # Check abbreviation (only for periods).
                    if ch == "." and self._is_abbreviation(i):
                        i = j
                        continue
                    return j

            i += 1

        return None

    def _is_abbreviation(self, dot_pos: int) -> bool:
        """Check if the period at *dot_pos* is part of an abbreviation."""
        buf = self._buffer

        # Walk backwards to find the word immediately before the period.
        end = dot_pos
        start = end - 1
        while start >= 0 and buf[start].isalpha():
            start -= 1
        start += 1

        if start >= end:
            return False

        word = buf[start:end]

        # Single letter (initial): "U." in "U.S.", "J." in "J. K. Rowling"
        if len(word) == 1 and word.isupper():
            return True

        # Known abbreviation list.
        return word.lower() in self._ABBREVIATIONS
