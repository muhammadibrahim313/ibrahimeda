import re

_space = re.compile(r"\s+")
_punct = re.compile(r"[^a-zA-Z\s]")

def basic_clean(s: str) -> str:
    s = s.lower()
    s = _punct.sub(" ", s)
    s = _space.sub(" ", s).strip()
    return s

def word_count(s: str) -> int:
    s = basic_clean(s)
    return 0 if not s else len(s.split(" "))
