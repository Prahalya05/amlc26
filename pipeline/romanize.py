"""Rough romanisation of Indic-script text (hand-written table, no external data).

Why: 7% of true pairs have the S1 name in Latin letters and the S2/S3 name in an Indian script
("jai finance limited" <-> "जय फाइनेंस लिमिटेड"). String similarity needs both in one alphabet.
skeleton.py keeps only consonant classes; this keeps vowels too, so rapidfuzz scores are comparable
with Latin-Latin scores:  जय फाइनेंस लिमिटेड -> "jay phainens limited".

The nine major Brahmic scripts (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu,
Kannada, Malayalam) share the same layout (blocks of 0x80 code points, letters at the same offset),
so one offset table covers all of them. Rules: consonants carry an inherent "a" unless followed by a
vowel sign or virama (the "halant"); the inherent "a" at the end of a word is dropped (Hindi-style
schwa deletion); long and short vowels are merged (aa -> a, ii -> i), as Latin spellings rarely mark them.
Non-Indic characters pass through unchanged.
"""
import re
import unicodedata

_CONS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh", 0x1C: "j",
         0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t",
         0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b",
         0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l",
         0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
         0x58: "k", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y"}
_VOW = {0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri", 0x0C: "li",
        0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au",
        0x60: "ri", 0x61: "li"}
_SIGN = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri", 0x45: "e",
         0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x62: "li",
         0x63: "li", 0x57: "au"}
_MOD = {0x01: "n", 0x02: "n", 0x03: "h", 0x70: "n"}  # 0x70 = Gurmukhi tippi (nasal)
_CHILLU = {0x7A: "n", 0x7B: "n", 0x7C: "r", 0x7D: "l", 0x7E: "l", 0x7F: "k"}  # Malayalam, no vowel
_VIRAMA, _NUKTA = 0x4D, 0x3C
_BRAHMIC = re.compile(r"[ऀ-ൿ]")


def _off(ch):
    cp = ord(ch)
    return cp & 0x7F if 0x0900 <= cp < 0x0D80 else None


def romanize(text: str) -> str:
    if not text or not _BRAHMIC.search(text):
        return text
    out = []
    # NFC re-joins two-part vowel signs that NFKD split (Bengali/Tamil/Malayalam "o" = "e" + "aa")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("റ്റ", "ട്ട")  # Malayalam rra-rra is "tt"
    text = text.replace("ന്റ", "ന്ട")  # Malayalam n-rra is "nt"
    pending_a = False  # a consonant was just written and still carries its inherent "a"
    for ch in text:
        o = _off(ch)
        if o is None:  # space, Latin letter, digit ...
            pending_a = False  # word end: inherent "a" dropped
            out.append(ch)
            continue
        if o == _NUKTA:
            continue
        if o in _CONS:
            if pending_a:
                out.append("a")
            out.append(_CONS[o])
            pending_a = True
        elif o in _SIGN:
            out.append(_SIGN[o])
            pending_a = False
        elif o == _VIRAMA:
            pending_a = False
        elif o in _VOW:
            if pending_a:
                out.append("a")
            out.append(_VOW[o])
            pending_a = False
        elif o in _CHILLU:
            if pending_a:
                out.append("a")
            out.append(_CHILLU[o])
            pending_a = False
        elif o in _MOD:
            if pending_a:
                out.append("a")
            out.append(_MOD[o])
            pending_a = False
        # anything else (digits are already ASCII after normalisation, avagraha ...) is dropped
    s = "".join(out)
    s = re.sub(r"aa+", "a", s)
    s = re.sub(r"ii+", "i", s)
    s = re.sub(r"uu+", "u", s)
    return s



# Romanised spellings of legal words ("प्राइवेट लिमिटेड" -> "praivet limited", "प्रा लि" -> "pra li"),
# removed like the Latin legal words in common.NAME_STOP when building the core name.
ROMAN_STOP = set("""
praivet praibhet praivatt piraivet privet pvt pra prai li lim limited limitet limittad limited
elaelapi elelpi elalpi llp kampani kampni kanpani company inc ltd end and
""".split())


def roman_core(core: str) -> str:
    from common import NAME_STOP
    return " ".join(w for w in romanize(core).split() if w not in ROMAN_STOP and w not in NAME_STOP)


if __name__ == "__main__":
    import os
    import unicodedata

    from common import WORK
    tests = ["राम मार्केटिंग प्राइवेट लिमिटेड", "जय फाइनेंस लिमिटेड", "குரு சர்வீசஸ் பிரைவேட் லிமிடெட்",
             "સિલ્વર ઇનોવેટિવ મેનેજમેન્ટ", "গুজরাট লোটাস ফুড প্রাইভেট লিমিটেড",
             "രാജ് ബാലാജി എന്റർപ്രൈസസ് പ്രൈവറ്റ് ലിമിറ്റഡ്", "ईस्टर्न बालाजी एंटरप्राइजेज प्राइवेट लिमिटेड"]
    with open(os.path.join(WORK, "reports", "romanize_test.txt"), "w", encoding="utf-8") as f:
        for t in tests:
            f.write(f"{t} -> {romanize(unicodedata.normalize('NFKD', t))}\n")
    print("written reports/romanize_test.txt")
