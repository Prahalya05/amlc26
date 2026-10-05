"""Shared paths, readers and text normalisation used by every pipeline step."""
import csv
import os
import sys
import time
import unicodedata

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

ROOT = os.environ.get("AMLC_ROOT", r"C:\amlc")
RAW = os.path.join(ROOT, "student_resource", "dataset")
WORK = os.environ.get("AMLC_WORK", os.path.join(ROOT, "pipeline"))
DATA = os.path.join(WORK, "data")
os.makedirs(DATA, exist_ok=True)

EXPECTED_ROWS = {
    ("train", 1): 2_206_821, ("train", 2): 5_034_616, ("train", 3): 5_285_603,
    ("test", 1): 1_732_544, ("test", 2): 4_887_273, ("test", 3): 5_082_316,
    ("train", "gt"): 2_206_821,
}

# Source prefix -> integer offset, so every record gets a unique int64 "rid".
SRC_OFFSET = {1: 1 * 10**10, 2: 2 * 10**10, 3: 3 * 10**10}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- RAM guard (Windows)
# Every script imports common, so every script logs its peak RAM at exit and aborts itself
# if its working set grows beyond AMLC_MAXMB (default 4300 MB) instead of starving the laptop.
import atexit
import ctypes
import threading
from ctypes import wintypes


class _PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


_K32 = ctypes.WinDLL("kernel32")
_K32.GetCurrentProcess.restype = ctypes.c_void_p  # 64-bit handle; the default int return type breaks it
_K32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD]


def mem_mb():
    """(current, peak) working set of this process in MB."""
    try:
        c = _PMC()
        c.cb = ctypes.sizeof(_PMC)
        _K32.K32GetProcessMemoryInfo(_K32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.WorkingSetSize / 2**20, c.PeakWorkingSetSize / 2**20
    except Exception:
        return 0.0, 0.0


MAX_MB = float(os.environ.get("AMLC_MAXMB", "4300"))


def _guard():
    while True:
        cur, _ = mem_mb()
        if cur > MAX_MB:
            print(time.strftime("%H:%M:%S"), f"RAM GUARD: working set {cur:.0f} MB > {MAX_MB:.0f} MB, aborting",
                  flush=True)
            os._exit(3)
        time.sleep(0.5)


threading.Thread(target=_guard, daemon=True).start()
atexit.register(lambda: print(time.strftime("%H:%M:%S"), f"peak RAM {mem_mb()[1]:.0f} MB", flush=True))


def raw_path(split, src):
    name = f"{split}_ground_truth.tsv" if src == "gt" else f"{split}_source{src}.tsv"
    return os.path.join(RAW, split, name)


def pq_path(split, src):
    return os.path.join(DATA, f"{split}_s{src}.parquet")


def read_tsv_chunks(path, chunksize=500_000):
    """The only way we read raw TSVs: tab separated, no quoting, nothing turned into NaN."""
    return pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, na_filter=False, dtype=str,
                       encoding="utf-8", chunksize=chunksize)


def id_to_rid(ids: pa.Array) -> pa.Array:
    """'S2-123' -> 2*10^10 + 123 (int64)."""
    src = pc.cast(pc.utf8_slice_codeunits(ids, 1, 2), pa.int64())
    num = pc.cast(pc.utf8_slice_codeunits(ids, 3), pa.int64())
    return pc.add(pc.multiply(src, 10**10), num)


def rid_to_id(rid: np.ndarray) -> np.ndarray:
    rid = np.asarray(rid, dtype=np.int64)
    src = rid // 10**10
    num = rid % 10**10
    return np.char.add(np.char.add("S", src.astype(str)), np.char.add("-", num.astype(str)))


# ---------------------------------------------------------------- normalisation
# Every Unicode decimal digit (Devanagari, Tamil, Arabic-Indic, full-width ...) -> ASCII 0-9.
_DIGITS = {cp: str(unicodedata.digit(chr(cp))) for cp in range(sys.maxunicode + 1)
           if unicodedata.category(chr(cp)) == "Nd" and not (48 <= cp <= 57)}
_DIGIT_TABLE = str.maketrans(_DIGITS)


def _ascii_digits(arr: pa.Array) -> pa.Array:
    """Python pass only over the rows that contain a non-ASCII character (a small minority)."""
    non_ascii = pc.invert(pc.string_is_ascii(arr)).to_numpy(zero_copy_only=False)
    if not non_ascii.any():
        return arr
    vals = arr.to_pylist()
    for i in np.flatnonzero(non_ascii):
        v = vals[i]
        if v is not None:
            vals[i] = v.translate(_DIGIT_TABLE)
    return pa.array(vals, type=pa.string())


def normalize(arr: pa.Array, kind: str) -> pa.Array:
    """NFKD, drop Latin accents, lowercase, & -> and, unicode digits -> ASCII,
    punctuation -> space, collapse spaces.  kind = 'name' | 'addr'.
    For names, apostrophes and dots are deleted (L.L.C. -> llc, Orelee's -> orelees);
    for addresses they become spaces (No.53 -> no 53)."""
    a = pc.utf8_normalize(arr, "NFKD")
    a = pc.replace_substring_regex(a, "[\u0300-\u036f]", "")  # combining Latin accents only
    a = pc.utf8_lower(a)
    a = pc.replace_substring(a, "&", " and ")
    if kind == "name":
        a = pc.replace_substring_regex(a, "['\u2019`.]", "")
    a = _ascii_digits(a)
    # keep letters, numbers and combining marks (Indic vowel signs are marks!)
    a = pc.replace_substring_regex(a, r"[^\p{L}\p{N}\p{M}]+", " ")
    a = pc.utf8_trim_whitespace(a)
    return a


# Street-type abbreviations: map both spellings to one canonical short form.
ADDR_ABBR = {
    "street": "st", "road": "rd", "avenue": "ave", "av": "ave", "drive": "dr", "lane": "ln",
    "court": "ct", "boulevard": "blvd", "highway": "hwy", "place": "pl", "circle": "cir",
    "terrace": "ter", "parkway": "pkwy", "square": "sq", "trail": "trl", "mount": "mt",
    "north": "n", "south": "s", "east": "e", "west": "w", "suite": "ste", "apartment": "apt",
    "floor": "fl", "building": "bldg", "number": "no", "house": "h", "sector": "sec",
    "near": "nr", "opposite": "opp", "cross": "crs", "main": "mn", "nagar": "ngr",
    "fort": "ft", "saint": "st", "route": "rte", "unit": "unit", "first": "1st",
    "second": "2nd", "third": "3rd", "ground": "gr", "gf": "gr",
}


def canon_addr(arr: pa.Array) -> pa.Array:
    a = arr
    for long, short in ADDR_ABBR.items():
        if long != short:
            a = pc.replace_substring_regex(a, rf"\b{long}\b", short)
    return a


def addr_numbers(addr_n: pa.Array) -> pa.Array:
    """All digit runs in the normalised address, leading zeros removed, space separated."""
    a = pc.replace_substring_regex(addr_n, r"[^0-9]+", " ")
    a = pc.utf8_trim_whitespace(a)
    a = pc.replace_substring_regex(a, r"(^| )0+([0-9])", r"\1\2")
    return a


# Generic legal / filler words removed to get the "core" of a business name.
NAME_STOP = set("""
llc inc incorporated ltd limited pvt private co company corp corporation the and of llp pc
lp plc pllc l l c pty group sa sas sarl sci eurl snc gmbh dba ag holdings public opc
""".split())


def name_core(name_n: pa.Array) -> pa.Array:
    pat = r"\b(" + "|".join(sorted(NAME_STOP, key=len, reverse=True)) + r")\b"
    a = pc.replace_substring_regex(name_n, pat, " ")
    a = pc.replace_substring_regex(a, r"\s+", " ")
    return pc.utf8_trim_whitespace(a)


def script_flags(arr: pa.Array):
    deva = pc.match_substring_regex(arr, r"[\x{0900}-\x{097F}]")
    nonlatin = pc.match_substring_regex(arr, r"[\x{0370}-\x{1FFF}\x{3000}-\x{FFFF}]")
    return deva, nonlatin
