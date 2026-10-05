"""Script-independent phonetic "consonant skeleton" of a name.

Why: some Source-2/3 names are written in an Indian script (e.g. Devanagari "राम मार्केटिंग
प्राइवेट लिमिटेड") while Source 1 has the Latin spelling ("Ram Marketing Private Limited").
We map BOTH to a coarse consonant code so they become comparable:
    ram marketing private limited  -> RN NRKTNK PRPT LNTT
    राम मार्केटिंग प्राइवेट लिमिटेड     -> RN NRKTNK PRPT LNTT
Rules (hand-written, no external data): vowels, h and y are dropped; similar-sounding consonants
share one class: K = k g q c(hard) x ; T = t d ; P = p b f v w ; S = s z j c(soft) ch sh ;
N = n m (and the nasal marks) ; L = l ; R = r. Repeated letters collapse.
All nine major Brahmic scripts (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu,
Kannada, Malayalam) share the same Unicode layout (each block is 0x80 wide and letters sit at
the same offset), so a single offset table covers all of them.
"""
import re

# offsets inside a Brahmic block -> class letter ('' = drop)
_OFF = {}
for off, c in {
    0x01: "N", 0x02: "N",  # candrabindu, anusvara
    0x15: "K", 0x16: "K", 0x17: "K", 0x18: "K", 0x19: "N",
    0x1A: "S", 0x1B: "S", 0x1C: "S", 0x1D: "S", 0x1E: "N",
    0x1F: "T", 0x20: "T", 0x21: "T", 0x22: "T", 0x23: "N",
    0x24: "T", 0x25: "T", 0x26: "T", 0x27: "T", 0x28: "N", 0x29: "N",
    0x2A: "P", 0x2B: "P", 0x2C: "P", 0x2D: "P", 0x2E: "N",
    0x2F: "", 0x30: "R", 0x31: "R", 0x32: "L", 0x33: "L", 0x34: "L", 0x35: "P",
    0x36: "S", 0x37: "S", 0x38: "S", 0x39: "",
    0x58: "K", 0x59: "K", 0x5A: "K", 0x5B: "S", 0x5C: "T", 0x5D: "T", 0x5E: "P", 0x5F: "",
}.items():
    _OFF[off] = c

_TABLE = {}
for base in range(0x0900, 0x0D80, 0x80):
    for off in range(0x80):
        cp = base + off
        _TABLE[cp] = _OFF.get(off, "")  # vowels, virama, nukta, digits(already ASCII) -> drop
# Latin: handle digraphs first, then single letters
_LAT_DI = [("tion", "sn"), ("ch", "S"), ("sh", "S"), ("ph", "P"), ("th", "T"), ("kh", "K"), ("gh", "K"),
           ("bh", "P"), ("dh", "T"), ("ck", "K"), ("x", "KS"), ("qu", "K")]
_LAT = {"k": "K", "g": "K", "q": "K", "c": "K", "t": "T", "d": "T", "p": "P", "b": "P",
        "f": "P", "v": "P", "w": "P", "s": "S", "z": "S", "j": "S", "n": "N", "m": "N",
        "l": "L", "r": "R"}
_SOFT_C = re.compile(r"c(?=[eiy])")
_BRAHMIC = re.compile(r"[ऀ-ൿ]")


def skel_token(tok: str) -> str:
    if _BRAHMIC.search(tok):
        s = "".join(_TABLE.get(ord(ch), "") if 0x0900 <= ord(ch) < 0x0D80 else _LAT.get(ch, "")
                    for ch in tok)
    else:
        t = _SOFT_C.sub("s", tok)
        for a, b in _LAT_DI:
            t = t.replace(a, b.lower() if b != "KS" else "ks")
        s = "".join(_LAT.get(ch, ch if ch.isdigit() else "") for ch in t)
    # collapse repeats
    return re.sub(r"(.)\1+", r"\1", s)


def skeleton(name_n: str) -> list:
    return [s for s in (skel_token(t) for t in name_n.split()) if s]


# skeletons of generic words (legal suffixes etc.) are removed from the "core" skeleton
_GENERIC = ("llc inc incorporated ltd limited pvt private co company corp corporation the and of "
            "llp pc lp plc pllc pty group sa sas sarl sci eurl dba ag holdings public opc "
            "प्राइवेट लिमिटेड प्रा लि एलएलपी").split()
# one-letter skeletons (the -> T, co -> K ...) are too ambiguous to remove
GENERIC_SKEL = {s for s in (skel_token(w) for w in _GENERIC) if len(s) >= 2}


def skeleton_core(name_n: str) -> str:
    sk = [s for s in skeleton(name_n) if s not in GENERIC_SKEL]
    return " ".join(sk)


if __name__ == "__main__":
    tests = [("ram marketing private limited", "राम मार्केटिंग प्राइवेट लिमिटेड"),
             ("global business private limited", "குளோபல் பிசினஸ் பிரைவேட் லிமிடெட்"),
             ("shakti urban products private limited", "શક્તિ અર્બન પ્રોડક્ટ્સ પ્રાઇવેટ લિમિટેડ"),
             ("aditya properties llp", "आदित्य प्रॉपर्टीज एलएलपी"),
             ("sun constructions private limited", "सन कंस्ट्रक्शंस प्राइवेट लिमिटेड")]
    import unicodedata
    import os
    from common import DATA
    with open(os.path.join(DATA, "skeleton_test.txt"), "w", encoding="utf-8") as f:
        ok = 0
        for a, b in tests:
            b = unicodedata.normalize("NFKD", b)
            sa, sb = skeleton_core(a), skeleton_core(b)
            ok += sa == sb
            f.write(f"{a} -> {sa}\n{b} -> {sb}\n\n")
    print(f"{ok}/{len(tests)} exact skeleton agreements (details in skeleton_test.txt)")
