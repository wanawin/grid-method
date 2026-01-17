# pick3_touch_grid_app_v14.py
# Streamlit app: Pick 3 Due-Digit Grid (touching pairs) + Parity Pair Chart strategy
# Produces FULL qualifying BOX list ranked, plus a walk-forward backtest to compare methods per state / draw stream.
#
# Notes:
# - BOX logic throughout (sorted digits). 013 represents any permutation of 0,1,3.
# - No look-ahead: backtest builds candidates using only prior draws.
#
# Supported inputs: LotteryPost-style tab-delimited TXT lines, with optional "Fireball" text after the number.
# Example:
#   Sat, Jan 10, 2026\tTexas\tPick 3 Day\t3-6-2, Fireball: 7

import re
import itertools
import datetime as dt
from pathlib import Path

from collections import Counter, defaultdict

import pandas as pd

def _safe_date_str(date_series, which: str = 'min') -> str:
    """Return YYYY-MM-DD for min/max of a date-like Series, robust to strings/NaT."""
    try:
        if date_series is None:
            return ''
        s = pd.to_datetime(date_series, errors='coerce')
        if getattr(s, 'empty', False):
            return ''
        val = s.min() if which == 'min' else s.max()
        if pd.isna(val):
            return ''
        # val is Timestamp
        return str(val.date())
    except Exception:
        return ''


import numpy as np
import streamlit as st

DIGITS = [str(i) for i in range(10)]
DRAW_ORDER = {"Morning": 0, "Midday": 1, "Day": 1, "Evening": 2, "Night": 3}

# ---------------------------
# Parsing
# ---------------------------

# Matches both "Day" and "Midday" labels.
_LOTTERYPOST_PAT = re.compile(
    r'^(?P<date>[A-Za-z]{3},\s+[A-Za-z]{3}\s+\d{1,2},\s+\d{4})\t'
    r'(?P<state>[^\t]+)\t'
    r'Pick\s*3\s+(?P<draw>Morning|Midday|Day|Evening|Night)\t'
    r'(?P<num>\d-\d-\d)',
    re.IGNORECASE
)

def parse_history_text(raw: str) -> pd.DataFrame:
    """
    Parse LotteryPost-like tab-delimited TXT content.
    Returns DataFrame columns: date (date), draw, num (string 'XYZ'), state
    """
    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _LOTTERYPOST_PAT.search(line)
        if not m:
            continue
        date_str = m.group("date")
        state = m.group("state").strip()
        draw = m.group("draw").title()
        # normalize Midday->Day to keep 4-stream logic consistent
        if draw == "Midday":
            draw = "Day"
        num = m.group("num").replace("-", "")
        try:
            d = dt.datetime.strptime(date_str, "%a, %b %d, %Y").date()
        except Exception:
            continue
        if len(num) == 3 and num.isdigit():
            rows.append((d, draw, num, state))
    df = pd.DataFrame(rows, columns=["date", "draw", "num", "state"])
    if df.empty:
        return df
    df["draw_order"] = df["draw"].map(DRAW_ORDER).fillna(99).astype(int)
    df = df.sort_values(["state", "date", "draw_order"]).reset_index(drop=True)
    return df

# ---------------------------
# Helpers
# ---------------------------

def box_key(num: str) -> str:
    return "".join(sorted(list(num), key=int))


def box_of(num) -> int:
    """Return canonical box id (sorted digits) as an int.

    Examples:
      - 582 -> 258
      - "074" -> 47   (canonical box is "047")

    This is used for winner matching during backtests.
    """
    s = str(num).strip()
    # Keep only digits and pad to 3 so we handle values like 7 -> "007".
    s = "".join([c for c in s if c.isdigit()]).zfill(3)[:3]
    return int(box_key(s))

def structure(box: str) -> str:
    a, b, c = box
    if a == b == c:
        return "Triple"
    if a == b or b == c:
        return "Double"
    return "Single"

def parity_sig(num: str) -> str:
    ev = sum((int(d) % 2 == 0) for d in num)
    if ev == 3:
        return "EEE"
    if ev == 2:
        return "EEO"
    if ev == 1:
        return "EOO"
    return "OOO"

def neighbors(prev_digits: set[str], wrap_mod10: bool) -> set[str]:
    out = set()
    for ch in prev_digits:
        d = int(ch)
        if wrap_mod10:
            out.add(str((d - 1) % 10))
            out.add(str((d + 1) % 10))
        else:
            if d - 1 >= 0:
                out.add(str(d - 1))
            if d + 1 <= 9:
                out.add(str(d + 1))
    return out

# ---------------------------
# Due grid (most overdue per position)
# ---------------------------

def compute_drought(stream_nums: list[str], upto_exclusive: int, window: int | None = None) -> list[dict[str, int]]:
    """
    drought[pos][digit] = draws since last seen for digit in position pos.
    If window is provided, compute drought within last `window` draws only.
    """
    if window is None:
        start = 0
    else:
        start = max(0, upto_exclusive - window)

    last_seen = [{d: None for d in DIGITS} for _ in range(3)]
    for i in range(start, upto_exclusive):
        rel = i - start
        s = stream_nums[i]
        for pos in range(3):
            last_seen[pos][s[pos]] = rel

    drought = []
    for pos in range(3):
        dct = {}
        for d in DIGITS:
            if last_seen[pos][d] is None:
                dct[d] = (upto_exclusive - start)
            else:
                dct[d] = (upto_exclusive - start - 1) - last_seen[pos][d]
        drought.append(dct)
    return drought

def build_due_grid(stream_nums: list[str], upto_exclusive: int, rows: int = 4, window: int | None = None):
    """
    Grid is rows x 3: columns are Hundreds, Tens, Ones.
    Each column contains the top `rows` most-overdue digits for that position.
    """
    drought = compute_drought(stream_nums, upto_exclusive, window=window)
    cols = []
    for pos in range(3):
        items = sorted(drought[pos].items(), key=lambda kv: (-kv[1], int(kv[0])))
        top = [d for d, _ in items[:rows]]
        top_sorted = sorted(top, key=int)
        cols.append(top_sorted)
    grid = [[cols[c][r] for c in range(3)] for r in range(rows)]
    return grid, drought

def digit_pair_strength(grid: list[list[str]], include_diagonal: bool = True):
    """
    Returns a function strength(a,b) = max touch weight among occurrences in grid.
    Orthogonal touch = 1.0, diagonal touch = 0.8 (optional).
    """
    locs = defaultdict(list)
    for r, row in enumerate(grid):
        for c, d in enumerate(row):
            locs[d].append((r, c))

    def strength(a: str, b: str) -> float:
        best = 0.0
        for (r1, c1) in locs.get(a, []):
            for (r2, c2) in locs.get(b, []):
                dr = abs(r1 - r2)
                dc = abs(c1 - c2)
                if dr == 0 and dc == 0 and a == b:
                    # same cell counts as "touch" for AA only if digit repeats in grid by virtue of appearing in two cols;
                    # here we treat exact same cell as no touch.
                    continue
                if (dr == 1 and dc == 0) or (dr == 0 and dc == 1):
                    best = max(best, 1.0)
                elif include_diagonal and dr == 1 and dc == 1:
                    best = max(best, 0.8)
        return best

    return strength

def grid_digits(grid: list[list[str]]) -> set[str]:
    return set(d for row in grid for d in row)

# ---------------------------
# Strategy A: Grid-touch candidates (BOX)
# ---------------------------

def generate_grid_candidates(
    grid: list[list[str]],
    strength_fn,
    prev_digits: set[str],
    neigh_digits: set[str],
    third_pool_mode: str
):
    """
    Requires at least one touching pair among grid digits.
    Third digit pool can be configured.
    Returns: dict box->metadata
    """
    gdigits = sorted(grid_digits(grid), key=int)

    if third_pool_mode == "Grid + Prev + ±1":
        third_pool = sorted(set(gdigits) | set(prev_digits) | set(neigh_digits), key=int)
    elif third_pool_mode == "Grid + Prev":
        third_pool = sorted(set(gdigits) | set(prev_digits), key=int)
    elif third_pool_mode == "Grid only":
        third_pool = list(gdigits)
    else:
        third_pool = DIGITS

    # all touching pairs among grid digits
    touch_pairs = []
    for a, b in itertools.combinations_with_replacement(gdigits, 2):
        s = strength_fn(a, b)
        if s > 0:
            touch_pairs.append((a, b, s))

    boxes = {}
    for a, b, s in touch_pairs:
        for c in third_pool:
            box = box_key(a + b + c)
            meta = boxes.get(box)
            third_src = []
            if c in gdigits: third_src.append("GRID")
            if c in prev_digits: third_src.append("PREV")
            if c in neigh_digits: third_src.append("±1")
            third_src = "+".join(third_src) if third_src else "OTHER"

            if meta is None:
                boxes[box] = {
                    "box": box,
                    "touch_strength": s,
                    "touch_pair": "".join(sorted([a, b], key=int)),
                    "third_digit": c,
                    "third_source": third_src,
                    "method": "Grid",
                }
            else:
                # keep the strongest touch; if tie, prefer richer third-source
                if s > meta["touch_strength"]:
                    meta.update({"touch_strength": s, "touch_pair": "".join(sorted([a, b], key=int)), "third_digit": c, "third_source": third_src})
                elif s == meta["touch_strength"]:
                    # prefer GRID+PREV over GRID, etc.
                    def src_score(src: str) -> int:
                        return (("GRID" in src) * 4) + (("PREV" in src) * 2) + (("±1" in src) * 1)
                    if src_score(third_src) > src_score(meta["third_source"]):
                        meta.update({"third_digit": c, "third_source": third_src})

    return boxes, third_pool

def rank_grid_boxes(boxes: dict, drought: list[dict[str, int]], prev_digits: set[str], neigh_digits: set[str], gdigits: set[str]) -> pd.DataFrame:
    rows = []
    for box, meta in boxes.items():
        ds = [drought[pos][box[pos]] for pos in range(3)]
        drought_sum = sum(ds)
        src = meta["third_source"]
        src_bonus = (("GRID" in src) * 6) + (("PREV" in src) * 3) + (("±1" in src) * 1)

        # small preference: box contains any carryover digit
        carry = len(set(box) & set(prev_digits))
        carry_bonus = carry * 0.5

        # final score: touch dominates, then "support", then drought_sum as mild
        score = meta["touch_strength"] * 100 + src_bonus * 4 + carry_bonus * 5 + drought_sum * 0.2

        # bucket label (ordered)
        if "GRID" in src and "PREV" in src:
            bucket = "A:Touch + 3rd(Grid+Prev)"
        elif "GRID" in src:
            bucket = "B:Touch + 3rd(Grid)"
        elif "PREV" in src:
            bucket = "C:Touch + 3rd(Prev)"
        elif "±1" in src:
            bucket = "D:Touch + 3rd(±1)"
        else:
            bucket = "E:Touch + 3rd(Other)"

        rows.append({
            "box": box,
            "structure": structure(box),
            "score": score,
            "method_bucket": bucket,
            "touch_strength": meta["touch_strength"],
            "touch_pair": meta["touch_pair"],
            "third_digit": meta["third_digit"],
            "third_source": meta["third_source"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["score", "box"], ascending=[False, True]).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df

# ---------------------------
# Strategy B: Parity Pair Chart candidates (BOX)
# ---------------------------

ODD_DIGITS = [d for d in DIGITS if int(d) % 2 == 1]
EVEN_DIGITS = [d for d in DIGITS if int(d) % 2 == 0]

EVEN_PAIRS = ["02","04","06","08","24","26","28","46","48","68"]
ODD_PAIRS  = ["13","15","17","19","35","37","39","57","59","79"]

def dominant_parity_pattern(last_nums: list[str]) -> str:
    counts = Counter(parity_sig(n) for n in last_nums)
    if not counts:
        return "EEO"
    # Prefer the 2+1 patterns if tie (because chart is built for them)
    best = sorted(counts.items(), key=lambda kv: (-kv[1], {"EEO":0,"EOO":1,"EEE":2,"OOO":3}.get(kv[0],9)))[0][0]
    return best

def pick_digits_by_freq(
    last_nums: list[str],
    parity: str,  # "even" or "odd"
    k: int,
    prefer: str,  # "Hot/Strong" or "Due/Weak"
    force_include_digits: set[str] | None = None
) -> list[str]:
    """
    Pick k digits of given parity by frequency within last_nums.
    prefer=Hot => highest counts first; Due => lowest counts first.
    force_include_digits: if provided, ensures those digits (of correct parity) are included if possible.
    """
    counts = Counter("".join(last_nums))
    pool = EVEN_DIGITS if parity == "even" else ODD_DIGITS
    items = [(d, counts.get(d, 0)) for d in pool]
    if prefer == "Hot/Strong":
        items = sorted(items, key=lambda kv: (-kv[1], int(kv[0])))
    else:
        items = sorted(items, key=lambda kv: (kv[1], int(kv[0])))
    chosen = [d for d, _ in items[:k]]

    if force_include_digits:
        for d in sorted(force_include_digits, key=int):
            if d not in pool:
                continue
            if d in chosen:
                continue
            # replace last element
            if len(chosen) < k:
                chosen.append(d)
            else:
                chosen[-1] = d
    # de-dup and re-trim
    chosen = list(dict.fromkeys(chosen))
    return chosen[:k]

def pair_frequency(last_nums: list[str], pair: str) -> int:
    a, b = pair[0], pair[1]
    ct = 0
    for n in last_nums:
        s = set(n)
        if a in s and b in s:
            ct += 1
    return ct

def generate_parity_chart_candidates(
    nums: list[str],
    idx_upto_exclusive: int,
    pattern_window: int,
    pair_k: int,
    third_k: int,
    pair_digit_prefer: str,
    third_digit_prefer: str,
    recency_include: bool,
    pair_strength_window: int,
):
    """
    Build the writer-style candidate set:
    1) Determine dominant parity pattern in last `pattern_window` draws (ending at idx-1).
    2) Choose 4 digits from the parity that supplies the pair; choose 4 digits from opposite parity for the 3rd digit.
    3) Create all pairs among pair digits, then cross with third digits.
    Returns: dict box->meta
    """
    if idx_upto_exclusive <= 1:
        return {}, {"pattern":"", "pair_parity":"", "third_parity":""}

    start = max(0, idx_upto_exclusive - pattern_window)
    last_nums = nums[start:idx_upto_exclusive]
    patt = dominant_parity_pattern(last_nums)

    # determine which parity supplies the PAIR for the 2+1 pattern
    if patt == "EEO":
        pair_parity, third_parity = "even", "odd"
        allowed_pairs = set(EVEN_PAIRS)
    elif patt == "EOO":
        pair_parity, third_parity = "odd", "even"
        allowed_pairs = set(ODD_PAIRS)
    elif patt == "EEE":
        # fallback: treat as even-pair mode (still generate 2+1), but mark pattern
        pair_parity, third_parity = "even", "odd"
        allowed_pairs = set(EVEN_PAIRS)
    else:  # OOO
        pair_parity, third_parity = "odd", "even"
        allowed_pairs = set(ODD_PAIRS)

    prev_num = nums[idx_upto_exclusive - 1]
    prev_digits = set(prev_num)

    force_pair = prev_digits if recency_include else set()
    force_third = prev_digits if recency_include else set()

    pair_digits = pick_digits_by_freq(last_nums, pair_parity, pair_k, pair_digit_prefer, force_include_digits=force_pair)
    third_digits = pick_digits_by_freq(last_nums, third_parity, third_k, third_digit_prefer, force_include_digits=force_third)

    # make pairs (unordered), but only keep those that are actual chart pairs (writer's rows)
    pairs = []
    for a, b in itertools.combinations(sorted(pair_digits, key=int), 2):
        p = "".join(sorted([a, b], key=int))
        if p in allowed_pairs:
            pairs.append(p)

    # pair strength measured over last pair_strength_window draws
    start2 = max(0, idx_upto_exclusive - pair_strength_window)
    strength_nums = nums[start2:idx_upto_exclusive]
    pair_strength = {p: pair_frequency(strength_nums, p) for p in pairs}

    boxes = {}
    for p in pairs:
        a, b = p[0], p[1]
        for c in third_digits:
            box = box_key(a + b + c)
            # meta: store best (highest pair strength; then third preference)
            meta = boxes.get(box)
            if meta is None:
                boxes[box] = {
                    "box": box,
                    "pattern": patt,
                    "pair": p,
                    "pair_strength": pair_strength.get(p, 0),
                    "third_digit": c,
                    "method": "Chart",
                }
            else:
                if pair_strength.get(p, 0) > meta["pair_strength"]:
                    meta.update({"pair": p, "pair_strength": pair_strength.get(p, 0), "third_digit": c})

    info = {
        "pattern": patt,
        "pair_parity": pair_parity,
        "third_parity": third_parity,
        "pair_digits": pair_digits,
        "third_digits": third_digits,
        "pairs": pairs,
    }
    return boxes, info

def rank_chart_boxes(
    boxes: dict,
    prev_digits: set[str],
    neigh_digits: set[str],
    gdigits: set[str],
):
    rows = []
    for box, meta in boxes.items():
        c = meta["third_digit"]
        third_src = []
        if c in gdigits: third_src.append("GRID")
        if c in prev_digits: third_src.append("PREV")
        if c in neigh_digits: third_src.append("±1")
        third_src = "+".join(third_src) if third_src else "OTHER"
        src_bonus = (("GRID" in third_src) * 6) + (("PREV" in third_src) * 3) + (("±1" in third_src) * 1)

        score = 100 + meta.get("pair_strength", 0) * 10 + src_bonus * 2

        if "GRID" in third_src and "PREV" in third_src:
            bucket = "A:Chart + 3rd(Grid+Prev)"
        elif "GRID" in third_src:
            bucket = "B:Chart + 3rd(Grid)"
        elif "PREV" in third_src:
            bucket = "C:Chart + 3rd(Prev)"
        elif "±1" in third_src:
            bucket = "D:Chart + 3rd(±1)"
        else:
            bucket = "E:Chart + 3rd(Other)"

        rows.append({
            "box": box,
            "structure": structure(box),
            "score": score,
            "method_bucket": bucket,
            "pattern": meta.get("pattern", ""),
            "pair": meta.get("pair", ""),
            "pair_strength": meta.get("pair_strength", 0),
            "third_digit": c,
            "third_source": third_src,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["score", "box"], ascending=[False, True]).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df

# ---------------------------
# Combined ranking (Union)
# ---------------------------

def combine_rankings(grid_df: pd.DataFrame, chart_df: pd.DataFrame, bonus_if_in_both: float = 25.0) -> pd.DataFrame:
    if grid_df.empty and chart_df.empty:
        return pd.DataFrame()
    # To avoid pandas merge errors from overlapping column names, only merge the
    # minimal, explicitly-prefixed columns we need for the combined table.
    g = grid_df[["box", "structure", "score", "method_bucket"]].copy() if not grid_df.empty else pd.DataFrame(columns=["box", "structure", "score", "method_bucket"])
    c = chart_df[["box", "structure", "score", "method_bucket"]].copy() if not chart_df.empty else pd.DataFrame(columns=["box", "structure", "score", "method_bucket"])

    g = g.rename(columns={"structure": "grid_structure", "score": "grid_score", "method_bucket": "grid_bucket"})
    c = c.rename(columns={"structure": "chart_structure", "score": "chart_score", "method_bucket": "chart_bucket"})

    merged = pd.merge(g, c, on="box", how="outer")

    merged["grid_score"] = merged["grid_score"].fillna(0.0)
    merged["chart_score"] = merged["chart_score"].fillna(0.0)
    merged["grid_bucket"] = merged["grid_bucket"].fillna("")
    merged["chart_bucket"] = merged["chart_bucket"].fillna("")

    merged["in_grid"] = merged["grid_score"] > 0
    merged["in_chart"] = merged["chart_score"] > 0
    merged["score"] = merged["grid_score"] + merged["chart_score"] + (bonus_if_in_both * (merged["in_grid"] & merged["in_chart"]))

    merged["structure"] = merged["grid_structure"].fillna(merged["chart_structure"])
    merged["method"] = merged.apply(
        lambda r: "Combined" if (r["in_grid"] and r["in_chart"]) else ("Grid" if r["in_grid"] else "Chart"),
        axis=1,
    )

    def pick_bucket(row):
        if row["in_grid"] and row["in_chart"]:
            return "A:Grid+Chart (both)"
        if row["in_grid"]:
            return f"B:{row['grid_bucket'] or 'Grid'}"
        return f"C:{row['chart_bucket'] or 'Chart'}"

    merged["method_bucket"] = merged.apply(pick_bucket, axis=1)

    out_cols = [
        "box",
        "structure",
        "method",
        "method_bucket",
        "score",
        "grid_score",
        "chart_score",
        "in_grid",
        "in_chart",
        "grid_bucket",
        "chart_bucket",
    ]
    out = merged[out_cols].sort_values(["score", "box"], ascending=[False, True]).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out

# ---------------------------
# Backtest comparison
# ---------------------------

def winner_rank_in(df_ranked: pd.DataFrame, winner_box: str) -> tuple[bool, int, float]:
    if df_ranked.empty:
        return (False, -1, 1.0)
    m = df_ranked.index[df_ranked["box"] == winner_box]
    if len(m) == 0:
        return (False, -1, 1.0)
    pos = int(m[0]) + 1
    pct = pos / len(df_ranked)
    return (True, pos, pct)

@st.cache_data(show_spinner=False)

# -------------------------
# Adaptive ranking (history-learned boosts)
# -------------------------


def digits_of(num, width: int = 3) -> list[int]:
    """Return the digits of a Pick-3 number as ints, preserving leading zeros.

    Accepts int (e.g., 24) or str (e.g., '024').
    If a longer numeric string is provided, the last `width` digits are used.
    """
    if num is None:
        return []

    if isinstance(num, int):
        s = f"{num:0{width}d}"[-width:]
    else:
        s = str(num).strip()
        # Keep only digits (defensive for stray whitespace/punctuation)
        s = ''.join(ch for ch in s if ch.isdigit())
        if s == '':
            return []
        if len(s) < width:
            s = s.zfill(width)
        elif len(s) > width:
            s = s[-width:]

    return [int(ch) for ch in s]


def structure_of(num) -> str:
    """Structure label for a Pick-3 number (BOX): Single / Double / Triple."""
    ds = digits_of(num)
    if len(ds) != 3:
        return "Unknown"
    u = len(set(ds))
    if u == 1:
        return "Triple"
    if u == 2:
        return "Double"
    return "Single"

def parity_box_sig(num: int) -> str:
    # Order-invariant parity signature for BOX combos (e.g., 2 even + 1 odd -> 'EEO')
    ds = digits_of(num)
    ev = sum(1 for d in ds if d % 2 == 0)
    od = 3 - ev
    return ("E" * ev) + ("O" * od)

def hml_box_sig(num: int, low_max: int = 3, high_min: int = 7) -> str:
    # Order-invariant H/M/L signature for BOX combos using L=0-3, M=4-6, H=7-9 by default.
    ds = digits_of(num)
    h = sum(1 for d in ds if d >= high_min)
    l = sum(1 for d in ds if d <= low_max)
    m = 3 - h - l
    return ("H" * h) + ("M" * m) + ("L" * l)

def sum_bucket_pick3(s: int) -> str:
    # Fixed buckets across all states (Pick 3 sums are 0-27)
    if s <= 6:
        return "0-6"
    if s <= 11:
        return "7-11"
    if s <= 16:
        return "12-16"
    if s <= 21:
        return "17-21"
    return "22-27"

def root9(s: int) -> int:
    # "Root" on a 1-9 cycle; keep 0 as 0
    if s == 0:
        return 0
    r = s % 9
    return 9 if r == 0 else r

def multiset_overlap_count(a: int, b: int) -> int:
    # Overlap count that respects repeats (min count per digit).
    ca = {str(d): 0 for d in range(10)}
    cb = {str(d): 0 for d in range(10)}
    for ch in f"{a:03d}":
        ca[ch] += 1
    for ch in f"{b:03d}":
        cb[ch] += 1
    return sum(min(ca[k], cb[k]) for k in ca)

def box_features(num: int, seed_num: int | None = None, low_max: int = 3, high_min: int = 7) -> dict:
    ds = digits_of(num)
    s = int(sum(ds))
    feat = {
        "box": f"{num:03d}",
        "structure": structure_of(num),
        "parity_box": parity_box_sig(num),
        "hml_box": hml_box_sig(num, low_max=low_max, high_min=high_min),
        "sum": s,
        "sum_parity": "Even" if s % 2 == 0 else "Odd",
        "sum_bucket": sum_bucket_pick3(s),
        "root": root9(s),
        "odd_cnt": sum(1 for d in ds if d % 2 == 1),
        "even_cnt": sum(1 for d in ds if d % 2 == 0),
        "hi_cnt": sum(1 for d in ds if d >= high_min),
        "lo_cnt": sum(1 for d in ds if d <= low_max),
        "mid_cnt": sum(1 for d in ds if low_max < d < high_min),
    }
    if seed_num is not None:
        feat["seed_overlap"] = multiset_overlap_count(seed_num, num)
    return feat

def build_transition_frame(df_stream: pd.DataFrame, low_max: int = 3, high_min: int = 7) -> pd.DataFrame:
    # Assumes df_stream is sorted chronologically.
    nums = df_stream["num"].tolist()
    if len(nums) < 2:
        return pd.DataFrame()

    rows = []
    for i in range(1, len(nums)):
        seed = int(nums[i - 1])
        nxt = int(nums[i])
        sf = box_features(seed, seed_num=None, low_max=low_max, high_min=high_min)
        nf = box_features(nxt, seed_num=None, low_max=low_max, high_min=high_min)
        rows.append({
            "seed_num": seed,
            "next_num": nxt,
            "seed_structure": sf["structure"],
            "next_structure": nf["structure"],
            "seed_parity": sf["parity_box"],
            "next_parity": nf["parity_box"],
            "seed_hml": sf["hml_box"],
            "next_hml": nf["hml_box"],
            "seed_sum": sf["sum"],
            "next_sum": nf["sum"],
            "seed_sum_parity": sf["sum_parity"],
            "next_sum_parity": nf["sum_parity"],
            "seed_sum_bucket": sf["sum_bucket"],
            "next_sum_bucket": nf["sum_bucket"],
            "seed_root": sf["root"],
            "next_root": nf["root"],
            "overlap_cnt": multiset_overlap_count(seed, nxt),
            "delta_sum": nf["sum"] - sf["sum"],
        })

    tdf = pd.DataFrame(rows)
    # Delta buckets (helps stability)
    def _delta_bucket(d: int) -> str:
        if d <= -6:
            return "<= -6"
        if d <= -2:
            return "-5 to -2"
        if d <= 1:
            return "-1 to +1"
        if d <= 5:
            return "+2 to +5"
        return ">= +6"
    tdf["delta_bucket"] = tdf["delta_sum"].apply(_delta_bucket)
    return tdf

def _dist(series: pd.Series) -> dict:
    vc = series.value_counts(dropna=False)
    total = float(vc.sum()) if len(vc) else 1.0
    return {k: float(v) / total for k, v in vc.items()}

def _uniform_prob(dist: dict) -> float:
    k = max(1, len(dist))
    return 1.0 / k

def compute_adaptive_profile(tdf: pd.DataFrame, seed_num: int, condition_on_seed: bool = True) -> dict:
    # Returns dict of distributions to use for boosting.
    if tdf.empty:
        return {}

    # Seed features for conditioning
    seed_feat = box_features(seed_num)
    sp = seed_feat["parity_box"]
    sh = seed_feat["hml_box"]
    ss = seed_feat["structure"]
    ssb = seed_feat["sum_bucket"]
    ssp = seed_feat["sum_parity"]
    sr = seed_feat["root"]

    profile = {
        "seed_feat": seed_feat,
        "overall": {
            "parity": _dist(tdf["next_parity"]),
            "hml": _dist(tdf["next_hml"]),
            "structure": _dist(tdf["next_structure"]),
            "sum_bucket": _dist(tdf["next_sum_bucket"]),
            "sum_parity": _dist(tdf["next_sum_parity"]),
            "root": _dist(tdf["next_root"]),
            "overlap": _dist(tdf["overlap_cnt"]),
            "delta": _dist(tdf["delta_bucket"]),
        },
        "cond": {}
    }

    if condition_on_seed:
        # condition each distribution on a single seed feature (stable + not too sparse)
        profile["cond"]["parity|seed_parity"] = _dist(tdf.loc[tdf["seed_parity"] == sp, "next_parity"]) if (tdf["seed_parity"] == sp).any() else {}
        profile["cond"]["hml|seed_hml"] = _dist(tdf.loc[tdf["seed_hml"] == sh, "next_hml"]) if (tdf["seed_hml"] == sh).any() else {}
        profile["cond"]["structure|seed_structure"] = _dist(tdf.loc[tdf["seed_structure"] == ss, "next_structure"]) if (tdf["seed_structure"] == ss).any() else {}
        profile["cond"]["sum_bucket|seed_sum_bucket"] = _dist(tdf.loc[tdf["seed_sum_bucket"] == ssb, "next_sum_bucket"]) if (tdf["seed_sum_bucket"] == ssb).any() else {}
        profile["cond"]["sum_parity|seed_sum_parity"] = _dist(tdf.loc[tdf["seed_sum_parity"] == ssp, "next_sum_parity"]) if (tdf["seed_sum_parity"] == ssp).any() else {}
        profile["cond"]["root|seed_root"] = _dist(tdf.loc[tdf["seed_root"] == sr, "next_root"]) if (tdf["seed_root"] == sr).any() else {}
        profile["cond"]["overlap|seed_structure"] = _dist(tdf.loc[tdf["seed_structure"] == ss, "overlap_cnt"]) if (tdf["seed_structure"] == ss).any() else {}
        profile["cond"]["delta|seed_sum_bucket"] = _dist(tdf.loc[tdf["seed_sum_bucket"] == ssb, "delta_bucket"]) if (tdf["seed_sum_bucket"] == ssb).any() else {}

    return profile

def adaptive_boost_for_candidate(profile: dict, cand_num: int, seed_num: int,
                                use_cond: bool,
                                w_parity: float, w_hml: float, w_structure: float,
                                w_sum_bucket: float, w_sum_parity: float, w_root: float,
                                w_overlap: float, w_delta: float,
                                boost_strength: float,
                                low_max: int = 3, high_min: int = 7) -> tuple[float, dict]:
    # Returns (boost, detail_dict)
    if not profile:
        return 0.0, {}

    cand = box_features(cand_num, seed_num=seed_num, low_max=low_max, high_min=high_min)
    ov = profile["overall"]
    cd = profile.get("cond", {})

    def _prob(dist: dict, val, fallback: float) -> float:
        if not dist:
            return fallback
        return float(dist.get(val, 0.0))

    def _score_component(overall_dist: dict, cond_dist: dict, val, weight: float) -> float:
        if weight <= 0:
            return 0.0
        p_overall = _prob(overall_dist, val, 0.0)
        u = _uniform_prob(overall_dist) if overall_dist else 0.0
        if use_cond and cond_dist:
            p_cond = _prob(cond_dist, val, 0.0)
            return weight * 100.0 * (p_cond - p_overall)
        # overall-only: compare to uniform (above-uniform gets +)
        return weight * 100.0 * (p_overall - u)

    details = {}

    details["parity"] = _score_component(ov["parity"], cd.get("parity|seed_parity", {}), cand["parity_box"], w_parity)
    details["hml"] = _score_component(ov["hml"], cd.get("hml|seed_hml", {}), cand["hml_box"], w_hml)
    details["structure"] = _score_component(ov["structure"], cd.get("structure|seed_structure", {}), cand["structure"], w_structure)
    details["sum_bucket"] = _score_component(ov["sum_bucket"], cd.get("sum_bucket|seed_sum_bucket", {}), cand["sum_bucket"], w_sum_bucket)
    details["sum_parity"] = _score_component(ov["sum_parity"], cd.get("sum_parity|seed_sum_parity", {}), cand["sum_parity"], w_sum_parity)
    details["root"] = _score_component(ov["root"], cd.get("root|seed_root", {}), cand["root"], w_root)
    details["overlap"] = _score_component(ov["overlap"], cd.get("overlap|seed_structure", {}), cand["seed_overlap"], w_overlap)

    # Delta needs seed_sum, so evaluate val for this candidate: delta_sum bucket vs seed
    seed_s = profile["seed_feat"]["sum"]
    d = cand["sum"] - seed_s
    if d <= -6:
        db = "<= -6"
    elif d <= -2:
        db = "-5 to -2"
    elif d <= 1:
        db = "-1 to +1"
    elif d <= 5:
        db = "+2 to +5"
    else:
        db = ">= +6"
    details["delta"] = _score_component(ov["delta"], cd.get("delta|seed_sum_bucket", {}), db, w_delta)

    raw = float(sum(details.values()))
    boost = raw * (boost_strength / 100.0)

    # small clamp so it doesn't overpower the base rank unless user cranks it
    boost = float(max(-60.0, min(60.0, boost)))
    return boost, {"boost_raw": raw, "delta_bucket": db, **cand, **details}

def apply_adaptive_boost_to_combined(combined_rank: pd.DataFrame,
                                    df_stream: pd.DataFrame,
                                    seed_num: int,
                                    condition_on_seed: bool,
                                    boost_strength: float,
                                    feature_weights: dict,
                                    low_max: int = 3,
                                    high_min: int = 7,
                                    learn_window: int | None = None) -> pd.DataFrame:
    # Build transitions from stream history
    if df_stream is None or df_stream.empty:
        return combined_rank

    # Use last N transitions if requested
    df_hist = df_stream.copy()
    if learn_window and learn_window > 0:
        # Need N+1 rows to get N transitions
        df_hist = df_hist.tail(int(learn_window) + 1)

    tdf = build_transition_frame(df_hist, low_max=low_max, high_min=high_min)
    if tdf.empty or len(tdf) < 20:
        # Too little data -> don't distort ranking
        return combined_rank

    profile = compute_adaptive_profile(tdf, seed_num=seed_num, condition_on_seed=condition_on_seed)

    w = {
        "parity": float(feature_weights.get("parity", 1.0)),
        "hml": float(feature_weights.get("hml", 1.0)),
        "structure": float(feature_weights.get("structure", 1.0)),
        "sum_bucket": float(feature_weights.get("sum_bucket", 1.0)),
        "sum_parity": float(feature_weights.get("sum_parity", 1.0)),
        "root": float(feature_weights.get("root", 1.0)),
        "overlap": float(feature_weights.get("overlap", 1.0)),
        "delta": float(feature_weights.get("delta", 1.0)),
    }

    cr = combined_rank.copy()
    boosts = []
    boost_details = []
    for b in cr["box"].astype(str).tolist():
        num = int(b)
        boost, det = adaptive_boost_for_candidate(
            profile=profile, cand_num=num, seed_num=seed_num, use_cond=condition_on_seed,
            w_parity=w["parity"], w_hml=w["hml"], w_structure=w["structure"],
            w_sum_bucket=w["sum_bucket"], w_sum_parity=w["sum_parity"], w_root=w["root"],
            w_overlap=w["overlap"], w_delta=w["delta"],
            boost_strength=boost_strength,
            low_max=low_max, high_min=high_min
        )
        boosts.append(boost)
        boost_details.append(det)

    cr["adaptive_boost"] = boosts
    cr["score_base"] = cr["score"]
    cr["score"] = cr["score_base"] + cr["adaptive_boost"]
    # Useful columns to display / filter
    # Extract key candidate descriptors
    cr["parity_box"] = [d.get("parity_box") for d in boost_details]
    cr["hml_box"] = [d.get("hml_box") for d in boost_details]
    cr["sum"] = [d.get("sum") for d in boost_details]
    cr["sum_bucket"] = [d.get("sum_bucket") for d in boost_details]
    cr["sum_parity"] = [d.get("sum_parity") for d in boost_details]
    cr["root"] = [d.get("root") for d in boost_details]
    cr["seed_overlap"] = [d.get("seed_overlap") for d in boost_details]
    cr["delta_bucket_vs_seed"] = [d.get("delta_bucket") for d in boost_details]
    # Store component boosts (optional)
    cr["boost_parity"] = [d.get("parity", 0.0) * (boost_strength / 100.0) for d in boost_details]
    cr["boost_hml"] = [d.get("hml", 0.0) * (boost_strength / 100.0) for d in boost_details]
    cr["boost_structure"] = [d.get("structure", 0.0) * (boost_strength / 100.0) for d in boost_details]
    cr["boost_sum_bucket"] = [d.get("sum_bucket", 0.0) * (boost_strength / 100.0) for d in boost_details]
    cr["boost_sum_parity"] = [d.get("sum_parity", 0.0) * (boost_strength / 100.0) for d in boost_details]
    cr["boost_root"] = [d.get("root", 0.0) * (boost_strength / 100.0) for d in boost_details]
    cr["boost_overlap"] = [d.get("overlap", 0.0) * (boost_strength / 100.0) for d in boost_details]
    cr["boost_delta"] = [d.get("delta", 0.0) * (boost_strength / 100.0) for d in boost_details]

    cr = cr.sort_values("score", ascending=False).reset_index(drop=True)
    cr["rank"] = np.arange(1, len(cr) + 1)
    return cr


def run_backtest(
    nums: list[str],
    dates: list,
    rows_n: int,
    include_diag: bool,
    third_pool_mode: str,
    wrap_mod10: bool,
    use_window: bool,
    drought_window: int,
    # chart params
    pattern_window: int,
    pair_k: int,
    third_k: int,
    pair_digit_prefer: str,
    third_digit_prefer: str,
    recency_include: bool,
    pair_strength_window: int,
    # backtest params
    max_tests: int,
    combined_bonus: float,
):
    n = len(nums)
    if n < 50:
        return pd.DataFrame(), pd.DataFrame()

    start_idx = max(20, pattern_window + 2)
    idxs = list(range(start_idx, n))
    if max_tests and len(idxs) > max_tests:
        idxs = idxs[-max_tests:]

    rows = []
    for t in idxs:
        prev = nums[t-1]
        prev_digits = set(prev)
        neigh = neighbors(prev_digits, wrap_mod10=wrap_mod10)

        grid, drought = build_due_grid(nums, t, rows=rows_n, window=(drought_window if use_window else None))
        strength_fn = digit_pair_strength(grid, include_diagonal=include_diag)
        gdigits = grid_digits(grid)

        grid_boxes, _ = generate_grid_candidates(grid, strength_fn, prev_digits, neigh, third_pool_mode)
        grid_rank = rank_grid_boxes(grid_boxes, drought, prev_digits, neigh, gdigits)

        chart_boxes, info = generate_parity_chart_candidates(
            nums=nums,
            idx_upto_exclusive=t,
            pattern_window=pattern_window,
            pair_k=pair_k,
            third_k=third_k,
            pair_digit_prefer=pair_digit_prefer,
            third_digit_prefer=third_digit_prefer,
            recency_include=recency_include,
            pair_strength_window=pair_strength_window,
        )
        chart_rank = rank_chart_boxes(chart_boxes, prev_digits, neigh, gdigits)
        comb_rank = combine_rankings(grid_rank, chart_rank, bonus_if_in_both=combined_bonus)

        winner = nums[t]
        wbox = box_key(winner)

        g_ok, g_pos, g_pct = winner_rank_in(grid_rank, wbox)
        c_ok, c_pos, c_pct = winner_rank_in(chart_rank, wbox)
        m_ok, m_pos, m_pct = winner_rank_in(comb_rank, wbox)

        rows.append({
            "date": dates[t],
            "winner": winner,
            "winner_box": wbox,
            "grid_n": len(grid_rank),
            "grid_hit": g_ok,
            "grid_rank": g_pos,
            "grid_pct": g_pct,
            "chart_n": len(chart_rank),
            "chart_hit": c_ok,
            "chart_rank": c_pos,
            "chart_pct": c_pct,
            "combined_n": len(comb_rank),
            "combined_hit": m_ok,
            "combined_rank": m_pos,
            "combined_pct": m_pct,
        })

    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()

    def summarize(prefix: str):
        hit = detail[f"{prefix}_hit"].mean()
        n_med = float(detail[f"{prefix}_n"].median())
        pct_avg = float(detail[f"{prefix}_pct"].mean())
        top20 = float((detail[f"{prefix}_rank"] <= 20).mean())
        top50 = float((detail[f"{prefix}_rank"] <= 50).mean())
        return {"method": prefix, "hit_rate": hit, "median_list_size": n_med, "avg_rank_pct": pct_avg, "top20_rate": top20, "top50_rate": top50}

    summary = pd.DataFrame([summarize("grid"), summarize("chart"), summarize("combined")])
    return detail, summary


# ---------------------------
# Percentile-zone utilities
# ---------------------------

def build_percentile_zone_table(detail: pd.DataFrame, prefix: str = "combined", bin_size: int = 5) -> pd.DataFrame:
    """Create a table of winner-rank percentiles binned into zones.

    Uses backtest detail columns like '{prefix}_hit' and '{prefix}_pct' where pct is rank/len (0..1).
    Lower percentile is better.
    """
    if detail is None or detail.empty:
        return pd.DataFrame()

    hit_col = f"{prefix}_hit"
    pct_col = f"{prefix}_pct"
    if hit_col not in detail.columns or pct_col not in detail.columns:
        return pd.DataFrame()

    d = detail[detail[hit_col] == True].copy()
    if d.empty:
        return pd.DataFrame()

    d["rank_pct"] = d[pct_col].astype(float) * 100.0
    d["rank_pct"] = d["rank_pct"].clip(lower=0.0, upper=100.0)

    d["bin_start"] = (d["rank_pct"] // bin_size).astype(int) * bin_size
    d.loc[d["bin_start"] >= 100, "bin_start"] = 100 - bin_size

    g = d.groupby("bin_start").size().reset_index(name="winner_hits")
    g["bin_end"] = g["bin_start"] + bin_size
    g = g.sort_values(["bin_start"]).reset_index(drop=True)

    total = float(g["winner_hits"].sum())
    g["pct_of_hits"] = g["winner_hits"] / total
    g["cum_pct"] = g["pct_of_hits"].cumsum()
    g["zone"] = g.apply(lambda r: f"{int(r['bin_start'])}-{int(r['bin_end'])}", axis=1)

    return g[["zone", "bin_start", "bin_end", "winner_hits", "pct_of_hits", "cum_pct"]]


def recommend_zones_for_target(zone_table: pd.DataFrame, target_coverage: float = 0.90) -> list[str]:
    """Pick zones that cover at least target_coverage of historical hits, choosing most-hit zones first."""
    if zone_table is None or zone_table.empty:
        return []

    z = zone_table.copy().sort_values(["winner_hits", "bin_start"], ascending=[False, True]).reset_index(drop=True)
    chosen: list[str] = []
    covered = 0.0
    for _, r in z.iterrows():
        chosen.append(str(r["zone"]))
        covered += float(r["pct_of_hits"])
        if covered >= target_coverage:
            break

    order = {str(r["zone"]): int(r["bin_start"]) for _, r in zone_table.iterrows()}
    return sorted(chosen, key=lambda x: order.get(x, 10**9))


def pad_ranked_to_universe(df_ranked: pd.DataFrame, universe_n: int = 1000) -> pd.DataFrame:
    """Pad a ranked df to a fixed universe size (e.g., 000–999) so percentiles are computed
    against a stable baseline, not just a small candidate list.

    Any missing boxes are added with a score lower than the current minimum score so they sort
    to the bottom. This makes percentile trimming behave more like your DC-5 model where the
    baseline universe is large.
    """
    if df_ranked is None or df_ranked.empty:
        return df_ranked

    if universe_n <= len(df_ranked):
        return df_ranked

    df = df_ranked.copy()

    # Determine a very-low score to push padded rows to the bottom.
    min_score = float(pd.to_numeric(df.get("score"), errors="coerce").min()) if "score" in df.columns else 0.0
    if pd.isna(min_score):
        min_score = 0.0
    pad_score = min_score - 1.0

    # Universe: 000..(universe_n-1) as 3-digit strings (default 1000 -> 000..999)
    universe_boxes = [f"{i:03d}" for i in range(universe_n)]
    have = set(df["box"].astype(str).str.zfill(3).tolist()) if "box" in df.columns else set()
    missing = [b for b in universe_boxes if b not in have]
    if not missing:
        return df

    cols = list(df.columns)
    rows = []
    for b in missing:
        r = {c: None for c in cols}
        r["box"] = b
        if "score" in cols:
            r["score"] = pad_score
        for bc in ("in_grid", "in_chart"):
            if bc in cols:
                r[bc] = False
        rows.append(r)

    df2 = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

    # Re-rank by score (desc) then box for stability
    if "score" in df2.columns:
        df2["_score_num"] = pd.to_numeric(df2["score"], errors="coerce").fillna(pad_score)
        df2 = df2.sort_values(by=["_score_num", "box"], ascending=[False, True]).drop(columns=["_score_num"])
    else:
        df2 = df2.sort_values(by=["box"], ascending=[True])

    df2 = df2.reset_index(drop=True)
    df2["rank"] = range(1, len(df2) + 1)
    return df2


def apply_percentile_trim(
    df_ranked: pd.DataFrame,
    mode: str,
    keep_top_pct: float,
    zones: list[str],
    bin_size: int = 5,
    percentile_basis: str = "Candidates",
    min_n_for_trim: int = 0,
) -> pd.DataFrame:
    """Trim a ranked list by percentile.

    mode:
      - 'Keep top %' keeps the top keep_top_pct percent of the list
      - 'Select zones' keeps only rows whose rank-percentile falls in selected zones

    Requires a 'rank' column on df_ranked.
    """
    if df_ranked is None or df_ranked.empty:
        return df_ranked

    df = df_ranked.copy()
    # Optionally compute percentiles against the full 000–999 universe (1000 boxes)
    # instead of the small candidate list. This makes trimming more stable and
    # far less likely to drop legit boxes just because the candidate list is small.
    if str(percentile_basis).lower().startswith("full"):
        df = pad_ranked_to_universe(df, universe_n=1000)
    n = len(df)
    if min_n_for_trim and n < int(min_n_for_trim):
        return df_ranked

    # Percentile of the ranked position.
    # Use (rank-1)/n so the best row is near 0% (not 0.03%) and bin edges behave more intuitively.
    # Example: with 100 rows, ranks 1..5 map to 0..4%, which cleanly fits the 0–5% zone.
    df["rank_pct"] = ((df["rank"].astype(float) - 1.0) / float(n)) * 100.0

    if mode == "Keep top %":
        k = max(1, int(round(n * (keep_top_pct / 100.0))))
        k = min(k, n)

        # Tie‑safe cutoff: if the ranking has a 'score' column, keep ALL rows tied with the last kept score.
        # This prevents edge cases where a winner falls exactly at the cutoff boundary.
        if "score" in df.columns:
            last_score = float(df.iloc[k - 1]["score"])
            kept = df[df["score"].astype(float) >= last_score].copy()
            return kept.reset_index(drop=True)

        return df.iloc[:k].reset_index(drop=True)

    if not zones:
        return df

    keep_ranges: list[tuple[float, float]] = []
    for z in zones:
        try:
            a, b = z.split("-")
            keep_ranges.append((float(a), float(b)))
        except Exception:
            continue

    def in_any(p: float) -> bool:
        # Zones are represented like "0-5", "5-10", ... and are meant to be half‑open [a,b)
        # so bins don't overlap. The (rank-1)/n percentile above ensures clean behavior at edges.
        for a, b in keep_ranges:
            if (p >= a) and (p < b):
                return True
        # Special‑case: if the user selected a zone ending in 100, include 100 exactly.
        if p >= 100.0:
            for a, b in keep_ranges:
                if b >= 100.0 and p >= a:
                    return True
        return False

    return df[df["rank_pct"].apply(in_any)].reset_index(drop=True)


# ---------------------------
# Transition-mode evaluation (Auto mode)
# ---------------------------

TRANSITION_SAME_DRAWTIME = "Same draw time (M->M)"
TRANSITION_PREV_OVERALL = "Previous overall draw (e.g., N->M, E->N)"

def _build_target_indices(df_state: pd.DataFrame, target_draw: str) -> list[int]:
    """Indices (positional) in df_state for rows matching target_draw (chronological order)."""
    if target_draw is None:
        return list(df_state.index)
    mask = df_state["draw_time"].astype(str).str.lower() == str(target_draw).lower()
    return list(df_state.index[mask])

def _slice_upto(df_state: pd.DataFrame, pos_exclusive: int) -> pd.DataFrame:
    """Return df_state rows strictly before positional pos_exclusive (0..len-1)."""
    return df_state.iloc[:pos_exclusive].copy()

def _positional_index_of(df_state: pd.DataFrame, df_index_value) -> int:
    """Map a df_state index value to its positional integer index."""
    return int(df_state.index.get_loc(df_index_value))

def run_transition_backtest(
    df_state: pd.DataFrame,
    target_draw: str,
    transition_mode: str,
    due_grid_window: int,
    chart_lookback: int,
    third_digit_sources: list[str],
    combined_bonus: float,
    max_tests: int = 250,
) -> pd.DataFrame:
    """Walk-forward backtest for a target draw time where the *seed* is chosen by transition_mode.

    Returns a per-test DataFrame containing:
      - winner (3-digit string)
      - winner_box (canonical box int)
      - winner_rank (1 = best) within the COMBINED ranking
      - winner_rank_pct (0 = best; (rank-1)/N), NaN if winner box not present
    """
    df_state = df_state.copy()
    if len(df_state) < 30:
        return pd.DataFrame()

    # Pull current UI settings (keeps behavior consistent with the main run)
    rows_n = int(st.session_state.get("rows_n", 4))
    include_diag = bool(st.session_state.get("include_diag", True))
    wrap_mod10 = bool(st.session_state.get("wrap_mod10", False))
    use_window = bool(st.session_state.get("use_window", True))

    # Chart settings
    pair_k = int(st.session_state.get("pair_k", 4))
    third_k = int(st.session_state.get("third_k", 4))
    pair_digit_prefer = str(st.session_state.get("pair_digit_prefer", "Hot/Strong"))
    third_digit_prefer = str(st.session_state.get("third_digit_prefer", "Due/Weak"))
    recency_include = bool(st.session_state.get("recency_include", True))
    pair_strength_window = int(st.session_state.get("pair_strength_window", 60))

    # Third pool mode comes from the multiselect (single choice)
    third_pool_mode = str(third_digit_sources[0]) if third_digit_sources else "Grid + Prev + ±1"

    # Positional target indices (chronological)
    target_idx_values = _build_target_indices(df_state, target_draw)
    target_positions = sorted([_positional_index_of(df_state, ix) for ix in target_idx_values])

    rows = []
    for pos in target_positions:
        if pos <= 0:
            continue

        # History available BEFORE this target draw
        df_hist = _slice_upto(df_state, pos)

        # Target-stream history for due-grid + chart
        df_target_hist = df_hist[df_hist["draw_time"].astype(str).str.lower() == str(target_draw).lower()]
        if len(df_target_hist) < max(5, chart_lookback):
            continue

        stream_nums = [str(x).strip().zfill(3) for x in df_target_hist["result"].astype(str).tolist()]
        if not stream_nums:
            continue

        # Winner (the current target draw)
        winner = str(df_state.iloc[pos]["result"]).strip().zfill(3)
        winner_box = box_of(winner)

        # Seed digits depend on the selected transition mode
        if str(transition_mode) == str(TRANSITION_PREV_OVERALL):
            seed_num = str(df_hist.iloc[-1]["result"]).strip().zfill(3)  # immediate previous draw overall
        else:
            seed_num = stream_nums[-1]  # previous draw in the same stream

        prev_digits = set(seed_num)
        neigh_digits = neighbors(prev_digits, wrap_mod10=wrap_mod10)

        # --- Due grid
        window = int(due_grid_window) if use_window else None
        grid, drought = build_due_grid(stream_nums, upto_exclusive=len(stream_nums), rows=rows_n, window=window)
        strength_fn = digit_pair_strength(grid, include_diagonal=include_diag)
        gdigits = grid_digits(grid)

        # --- Grid list
        grid_boxes, _ = generate_grid_candidates(grid, strength_fn, prev_digits, neigh_digits, third_pool_mode)
        grid_rank = rank_grid_boxes(grid_boxes, drought, prev_digits, neigh_digits, gdigits)

        # --- Chart list
        chart_boxes, _chart_info = generate_parity_chart_candidates(
            nums=stream_nums,
            idx_upto_exclusive=len(stream_nums),
            pattern_window=int(chart_lookback),
            pair_k=pair_k,
            third_k=third_k,
            pair_digit_prefer=pair_digit_prefer,
            third_digit_prefer=third_digit_prefer,
            recency_include=recency_include,
            pair_strength_window=pair_strength_window,
        )
        chart_rank = rank_chart_boxes(chart_boxes, prev_digits, neigh_digits, gdigits)

        # --- Combined
        comb_rank = combine_rankings(grid_rank, chart_rank, bonus_if_in_both=float(combined_bonus))

        # winner rank in combined list (box-based)
        w_ok, w_pos, w_pct = winner_rank_in(comb_rank, box_key(winner))

        rows.append(
            {
                "target_draw": str(target_draw),
                "transition_mode": str(transition_mode),
                "seed": seed_num,
                "winner": winner,
                "winner_box": winner_box,
                "combined_n": int(len(comb_rank)),
                "winner_rank": (int(w_pos) if w_ok else None),
                "winner_rank_pct": (float(w_pct) if w_ok else None),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    if max_tests and len(out) > max_tests:
        out = out.iloc[-max_tests:].reset_index(drop=True)
    return out

def summarize_transition_backtest(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {
            "n": 0,
            "avg_rank_pct": None,
            "median_rank_pct": None,
            "top10_rate": None,
            "top25_rate": None,
            "missing_rate": None,
        }
    n = len(df)
    avg_rank_pct = float(df["winner_rank_pct"].mean())
    median_rank_pct = float(df["winner_rank_pct"].median())
    top10_rate = float((df["winner_rank_pct"] <= 0.10).mean())
    top25_rate = float((df["winner_rank_pct"] <= 0.25).mean())
    missing_rate = float(df["winner_rank"].isna().mean())
    return {
        "n": int(n),
        "avg_rank_pct": avg_rank_pct,
        "median_rank_pct": median_rank_pct,
        "top10_rate": top10_rate,
        "top25_rate": top25_rate,
        "missing_rate": missing_rate,
    }

def choose_best_transition_mode(
    df_state: pd.DataFrame,
    target_draw: str,
    due_grid_window: int,
    chart_lookback: int,
    third_digit_sources: list[str],
    combined_bonus: float,
    learn_window: int,
) -> tuple[str, pd.DataFrame]:
    """Return (best_mode, summary_df)."""
    modes = [TRANSITION_SAME_DRAWTIME, TRANSITION_PREV_OVERALL]
    summaries = []

    for mode in modes:
        bt = run_transition_backtest(
            df_state=df_state,
            target_draw=target_draw,
            transition_mode=mode,
            due_grid_window=due_grid_window,
            chart_lookback=chart_lookback,
            third_digit_sources=third_digit_sources,
            combined_bonus=combined_bonus,
            max_tests=learn_window,
        )
        s = summarize_transition_backtest(bt)
        s["transition_mode"] = mode
        summaries.append(s)

    summary_df = pd.DataFrame(summaries)
    if summary_df.empty:
        return TRANSITION_SAME_DRAWTIME, summary_df

    def mode_score(row):
        if int(row.get("n", 0)) == 0:
            return -1e9
        return (
            2.0 * float(row["top25_rate"])
            + 1.0 * float(row["top10_rate"])
            - 1.0 * float(row["avg_rank_pct"])
            - 2.0 * float(row["missing_rate"])
        )

    summary_df["mode_score"] = summary_df.apply(mode_score, axis=1)
    best = summary_df.sort_values(["mode_score"], ascending=False).iloc[0]["transition_mode"]
    return str(best), summary_df.sort_values(["mode_score"], ascending=False).reset_index(drop=True)


# ---------------------------
# Calibration (auto settings per upload)
# ---------------------------

def _pick_recommended_from_calibration(df_cal: pd.DataFrame) -> dict:
    """Pick recommended setting with a 'hit-rate first' rule, then smallest list size."""
    if df_cal.empty:
        return {}
    # best hit
    best_hit = df_cal["hit_rate"].max()
    near_best = df_cal[df_cal["hit_rate"] >= (best_hit - 0.02)].copy()  # within 2%
    # prioritize: smaller median list, then better avg pct, then higher top50
    near_best = near_best.sort_values(
        ["median_list_size", "avg_rank_pct", "top50_rate", "top20_rate"],
        ascending=[True, True, False, False]
    )
    rec = near_best.iloc[0].to_dict()
    return rec

@st.cache_data(show_spinner=False)
def calibrate_grid_settings(
    nums: list[str],
    dates: list,
    rows_n: int,
    wrap_mod10: bool,
    # grid search space
    windows: tuple,
    diag_options: tuple,
    third_pool_options: tuple,
    # chart params (needed by run_backtest signature, kept fixed during calibration)
    pattern_window: int,
    pair_k: int,
    third_k: int,
    pair_digit_prefer: str,
    third_digit_prefer: str,
    recency_include: bool,
    pair_strength_window: int,
    # scoring
    max_tests: int,
    combined_bonus: float,
) -> tuple[pd.DataFrame, dict]:
    """
    Runs a walk-forward backtest for the GRID method across a small hyper-parameter grid and
    returns a calibration table + a recommended setting.

    NOTE: We score primarily on winner BOX inclusion (hit_rate), then prefer smaller list sizes.
    """
    if len(nums) < 60:
        return pd.DataFrame(), {}

    rows = []
    for w in windows:
        for diag in diag_options:
            for tpm in third_pool_options:
                detail, summary = run_backtest(
                    nums=nums,
                    dates=dates,
                    rows_n=rows_n,
                    include_diag=diag,
                    third_pool_mode=tpm,
                    wrap_mod10=wrap_mod10,
                    use_window=True,
                    drought_window=int(w),
                    pattern_window=pattern_window,
                    pair_k=pair_k,
                    third_k=third_k,
                    pair_digit_prefer=pair_digit_prefer,
                    third_digit_prefer=third_digit_prefer,
                    recency_include=recency_include,
                    pair_strength_window=pair_strength_window,
                    max_tests=max_tests,
                    combined_bonus=combined_bonus,
                )
                if summary.empty:
                    continue
                g = summary[summary["method"] == "grid"]
                if g.empty:
                    continue
                g = g.iloc[0].to_dict()

                # composite: hit_rate dominates; list size + avg rank pct are tie-breakers
                score = (g["hit_rate"] * 1000.0) - (g["median_list_size"] * 0.15) - (g["avg_rank_pct"] * 100.0)

                rows.append({
                    "window": int(w),
                    "include_diagonal": bool(diag),
                    "third_pool_mode": tpm,
                    "hit_rate": float(g["hit_rate"]),
                    "median_list_size": float(g["median_list_size"]),
                    "avg_rank_pct": float(g["avg_rank_pct"]),
                    "top20_rate": float(g["top20_rate"]),
                    "top50_rate": float(g["top50_rate"]),
                    "score": float(score),
                })

    df_cal = pd.DataFrame(rows)
    if df_cal.empty:
        return df_cal, {}

    df_cal = df_cal.sort_values(["hit_rate","median_list_size","avg_rank_pct","score"], ascending=[False, True, True, False]).reset_index(drop=True)
    rec = _pick_recommended_from_calibration(df_cal)
    return df_cal, rec

# ---------------------------
# UI
# ---------------------------

st.set_page_config(page_title="Pick 3 Grid + Parity Pair Chart", layout="wide")

due_grid_window = 6  # default lookback window for due-grid (can be overridden by sidebar slider)
st.title("Pick 3: Due-Grid Touch Pairs + Parity Pair Chart (FULL ranked boxes)")

# Session defaults (keeps widgets stable + lets calibration write values)
_DEFAULTS = {
    "rows_n": 4,
    "include_diag": True,
    "third_pool_mode": "Grid + Prev + ±1",
    "wrap_mod10": False,
    "use_window": False,
    "drought_window": 30,
    "pattern_window": 10,
    "pair_k": 4,
    "third_k": 4,
    "pair_digit_prefer": "Hot/Strong",
    "third_digit_prefer": "Due/Weak",
    "recency_include": True,
    "pair_strength_window": 60,
    "strategy_view": "Combined (union)",
    "combined_bonus": 25.0,
    "run_bt": False,
    "max_tests": 200,
    "cal_max_tests": 250,
    "auto_apply_cal": True,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

if "calibration_store" not in st.session_state:
    st.session_state["calibration_store"] = {}


with st.sidebar:
    st.header("1) Load history")
    uploaded = st.file_uploader("Upload LotteryPost TXT (tab-delimited)", type=["txt"], key="uploaded")
    st.caption("Order doesn't matter (newest→oldest or oldest→newest). The app sorts by date + draw time after parsing.")

    st.header("2) Choose State / Stream")
    stream_mode = st.selectbox("Stream mode", ["Per draw time", "All draws chronological"], index=0, key="stream_mode")

    # Default values (used when All draws chronological)
    draw = "All"
    transition_mode = TRANSITION_SAME_DRAWTIME
    learn_window = int(st.session_state.get("learn_window", 200))

    if stream_mode == "Per draw time":
        draw = st.selectbox("Draw time (if Per draw time)", ["Morning", "Day", "Evening", "Night"], index=0, key="draw_time")

        # How to choose the *seed* for this draw-time prediction
        transition_mode = st.selectbox(
            "Seed transition mode",
            ["Auto (choose best)", TRANSITION_SAME_DRAWTIME, TRANSITION_PREV_OVERALL],
            index=0,
            key="transition_mode",
            help="Auto evaluates which seed style ranks winners higher for this draw time (walk-forward on your uploaded history).",
        )

        learn_window = st.slider(
            "Auto-learn window (most recent tests)",
            min_value=50,
            max_value=500,
            value=int(learn_window),
            step=25,
            key="learn_window",
            help="How many most-recent walk-forward cases to use when Auto chooses the best transition mode.",
        )

    st.header("3) Grid strategy settings")

    # Due-grid lookback window (how many previous draws to compute positional due digits)
    due_grid_window = st.slider(
        "Due-grid lookback window (N draws back)",
        min_value=3,
        max_value=200,
        value=int(st.session_state.get("due_grid_window", 30)),
        step=1,
        help="For each position, treat digits missing from the last N draws (for that stream) as 'due'.",
        key="due_grid_window",
    )

    rows_n = st.slider("Grid rows (top overdue digits per position)", min_value=3, max_value=6, value=st.session_state.get("rows_n", 4), step=1, key="rows_n")
    include_diag = st.checkbox("Count diagonal touches (catacorner)", value=st.session_state.get("include_diag", True), key="include_diag")

    st.subheader("Third-digit pool (Grid strategy)")
    third_pool_mode = st.selectbox(
        "Third digit can come from:",
        ["Grid + Prev + ±1", "Grid + Prev", "Grid only", "All digits 0–9"],
        index=0,
        key="third_pool_mode"
    )

    # Backwards-compat alias used by auto-transition scorer
    third_digit_sources = [str(third_pool_mode)]
    wrap_mod10 = st.checkbox("±1 uses wrap-around (0↔9)", value=st.session_state.get("wrap_mod10", False), key="wrap_mod10")

    st.subheader("Due-grid lookback window (your tested method)")
    use_window = st.checkbox(
        "Use last-X-draws window to build the due-grid (recommended)",
        value=st.session_state.get("use_window", True),
        key="use_window",
        help="ON = due-grid is based on which digits are missing from the last X draws (per position). OFF = drought-style grid across all history to date.",
    )
    drought_window = st.slider(
        "X draws (lookback)",
        min_value=6,
        max_value=200,
        value=int(st.session_state.get("drought_window", 6)),
        step=1,
        disabled=not use_window,
        key="drought_window",
    )

    # Always define this for Auto transition selection + adaptive learning.
    # (Auto transition scoring assumes the last-X "missing in window" due-grid.)
    due_grid_window = int(drought_window)
    if not use_window:
        st.caption("Note: Auto transition scoring uses the last-X due-grid window. If you turn OFF windowing, Auto selection may not match your drought-style grid.")

    st.header("4) Parity Pair Chart settings")
    pattern_window = st.slider("Pattern window (N draws)", min_value=5, max_value=60, value=int(st.session_state.get("pattern_window", 10)), step=1, key="pattern_window")

    # Chart lookback is aligned to the parity-pattern lookback window
    chart_lookback = int(pattern_window)
    pair_k = st.slider("Pair-digit pool size (Kpair)", min_value=3, max_value=5, value=int(st.session_state.get("pair_k", 4)), step=1, key="pair_k")
    third_k = st.slider("Third-digit pool size (Kthird)", min_value=2, max_value=5, value=int(st.session_state.get("third_k", 4)), step=1, key="third_k")

    pair_digit_prefer = st.selectbox("Pair-digit preference", ["Hot/Strong", "Due/Weak"], index=0, key="pair_digit_prefer")
    third_digit_prefer = st.selectbox("Third-digit preference (toggle)", ["Hot/Strong", "Due/Weak"], index=1, key="third_digit_prefer")

    recency_include = st.checkbox("Recency include (force include prev-draw digits if possible)", value=st.session_state.get("recency_include", True), key="recency_include")
    pair_strength_window = st.slider("Pair-strength window (draws)", min_value=10, max_value=200, value=int(st.session_state.get("pair_strength_window", 60)), step=5, key="pair_strength_window")

    st.header("5) Output / Comparison")
    strategy_view = st.selectbox("Show ranked list for:", ["Grid only", "Chart only", "Combined (union)"], index=2, key="strategy_view")
    combined_bonus = st.slider("Combined bonus if box is in BOTH lists", min_value=0.0, max_value=100.0, value=float(st.session_state.get("combined_bonus", 25.0)), step=5.0, key="combined_bonus")

    st.header("6) Adaptive ranking (history-learned)")
    use_adaptive = st.checkbox(
        "Boost **Combined** ranking using history-learned patterns",
        value=bool(st.session_state.get("use_adaptive", True)),
        key="use_adaptive"
    )
    condition_on_seed = st.checkbox(
        "Condition boosts on the current seed (recommended)",
        value=bool(st.session_state.get("condition_on_seed", True)),
        disabled=not use_adaptive,
        key="condition_on_seed"
    )
    boost_strength = st.slider(
        "Boost strength (how much history can move the list)",
        0.0, 100.0,
        float(st.session_state.get("boost_strength", 35.0)),
        step=5.0,
        disabled=not use_adaptive,
        key="boost_strength"
    )
    learn_window_choice = st.selectbox(
        "Learning window (how far back to learn patterns)",
        ["All available history", "Last 200", "Last 500", "Last 1000"],
        index=0,
        disabled=not use_adaptive,
        key="learn_window_choice"
    )
    learn_window = None
    if learn_window_choice.startswith("Last"):
        try:
            learn_window = int(learn_window_choice.split()[1])
        except Exception:
            learn_window = None

    with st.expander("Advanced: per-feature weights", expanded=False):
        st.caption("Higher weight = that feature influences ranking more. Set to 0 to ignore a feature.")
        w_parity = st.slider("Parity pattern weight (EEO / OOE / EEE / OOO)", 0.0, 3.0, float(st.session_state.get("w_parity", 1.25)), step=0.25, disabled=not use_adaptive, key="w_parity")
        w_hml = st.slider("H/M/L pattern weight (H=7–9, M=4–6, L=0–3)", 0.0, 3.0, float(st.session_state.get("w_hml", 1.00)), step=0.25, disabled=not use_adaptive, key="w_hml")
        w_structure = st.slider("Structure weight (Single / Double / Triple)", 0.0, 3.0, float(st.session_state.get("w_structure", 1.00)), step=0.25, disabled=not use_adaptive, key="w_structure")
        w_sum_bucket = st.slider("Sum bucket weight (0–6 / 7–11 / 12–16 / 17–21 / 22–27)", 0.0, 3.0, float(st.session_state.get("w_sum_bucket", 0.75)), step=0.25, disabled=not use_adaptive, key="w_sum_bucket")
        w_sum_parity = st.slider("Sum parity weight (Odd/Even)", 0.0, 3.0, float(st.session_state.get("w_sum_parity", 0.50)), step=0.25, disabled=not use_adaptive, key="w_sum_parity")
        w_root = st.slider("Root (mod-9) weight", 0.0, 3.0, float(st.session_state.get("w_root", 0.25)), step=0.25, disabled=not use_adaptive, key="w_root")
        w_overlap = st.slider("Seed overlap weight (# digits carried over)", 0.0, 3.0, float(st.session_state.get("w_overlap", 1.00)), step=0.25, disabled=not use_adaptive, key="w_overlap")
        w_delta = st.slider("Sum-change weight (vs seed)", 0.0, 3.0, float(st.session_state.get("w_delta", 0.75)), step=0.25, disabled=not use_adaptive, key="w_delta")

    st.subheader("Percentile filter (optional)")
    use_pct_trim = st.checkbox("Enable percentile filter (trim ranked list)", value=bool(st.session_state.get("use_pct_trim", False)), key="use_pct_trim")
    pct_mode = st.selectbox("Percentile mode", ["Keep top %", "Select zones"], index=0, disabled=not use_pct_trim, key="pct_mode")
    keep_top_pct = st.slider("Keep top % (simple)", min_value=1, max_value=100, value=int(st.session_state.get("keep_top_pct", 90)), step=1, disabled=not (use_pct_trim and pct_mode=="Keep top %"), key="keep_top_pct")
    pct_basis = st.selectbox("Percentile basis", ["Full universe (000–999)", "Candidates (current list)"], index=0, disabled=not use_pct_trim, key="pct_basis")
    min_n_for_trim = st.slider("Skip trim if list smaller than", min_value=0, max_value=1000, value=int(st.session_state.get("min_n_for_trim", 0)), step=25, disabled=not use_pct_trim, key="min_n_for_trim")

    # --- Select-zones controls (robust to zone-size changes) ---
    zone_bin_size = st.selectbox("Zone size (for Select zones)", [1, 2, 5, 10], index=2, disabled=not (use_pct_trim and pct_mode=="Select zones"), key="zone_bin_size")
    zone_target_cov = st.slider("Auto-zone target coverage (needs backtest)", min_value=0.50, max_value=0.99, value=float(st.session_state.get("zone_target_cov", 0.90)), step=0.01, disabled=not (use_pct_trim and pct_mode=="Select zones"), key="zone_target_cov")
    zone_auto_pick_btn = st.button("Auto-pick zones from backtest", disabled=not (use_pct_trim and pct_mode=="Select zones"), key="zone_auto_pick_btn")
    all_zones = [f"{i}-{i+int(zone_bin_size)}" for i in range(0, 100, int(zone_bin_size))]

    # Quick way to 'keep 1 zone' (or any N) without hunting the multiselect:
    zones_keep_n_default = int(st.session_state.get("zones_keep_n", max(1, len(all_zones)//2)))
    zones_keep_n_default = max(1, min(len(all_zones), zones_keep_n_default))
    zones_keep_n = st.slider("Keep N zones (best ranks)", min_value=1, max_value=max(1, len(all_zones)), value=zones_keep_n_default, step=1, disabled=not (use_pct_trim and pct_mode=="Select zones"), key="zones_keep_n")
    auto_zones_from_n = st.checkbox("Auto-set zones to best N (recommended)", value=bool(st.session_state.get("auto_zones_from_n", True)), disabled=not (use_pct_trim and pct_mode=="Select zones"), key="auto_zones_from_n")

    # Sanitize previously saved zone selections when the zone size changes.
    saved_keep = st.session_state.get("zones_keep", [])
    saved_keep = [z for z in saved_keep if z in all_zones]
    if auto_zones_from_n:
        saved_keep = all_zones[:zones_keep_n]
        st.session_state["zones_keep"] = saved_keep
    if not saved_keep:
        saved_keep = all_zones[:zones_keep_n]
        st.session_state["zones_keep"] = saved_keep

    zones_keep = st.multiselect("Zones to keep (0 is best ranks)", options=all_zones, default=saved_keep, disabled=not (use_pct_trim and pct_mode=="Select zones"), key="zones_keep")


    st.subheader("Backtest")
    run_bt = st.checkbox("Run method comparison backtest", value=st.session_state.get("run_bt", False), key="run_bt")
    max_tests = st.slider("How many most-recent transitions to test", min_value=50, max_value=500, value=int(st.session_state.get("max_tests", 200)), step=25, disabled=not run_bt, key="max_tests")

    st.header("6) Auto-calibrate (recommended)")
    st.write("This runs a **walk-forward** backtest over a small set of grid settings and then auto-picks the best combo for your selected stream.")
    cal_windows = st.multiselect("Grid lookback candidates (X draws)", options=[6,7,8,9,10], default=[6,7,8,9,10], key="cal_windows")
    cal_try_diag = st.checkbox("Try diagonal ON and OFF", value=True, key="cal_try_diag")
    cal_third_pool = st.multiselect(
        "Try third-digit pools",
        options=["Grid + Prev + ±1", "Grid + Prev", "Grid only"],
        default=["Grid + Prev + ±1", "Grid + Prev", "Grid only"],
        key="cal_third_pool"
    )
    cal_max_tests = st.slider("Calibration test depth (most-recent transitions)", min_value=80, max_value=500, value=int(st.session_state.get("cal_max_tests", 250)), step=10, key="cal_max_tests")
    calibrate_now = st.button("Run calibration now", key="calibrate_now", disabled=(uploaded is None))
    auto_apply_cal = st.checkbox("Auto-apply saved calibration when available", value=True, key="auto_apply_cal")
    clear_cal = st.button("Clear saved calibrations", key="clear_cal")

# Load data
if not uploaded:
    st.info("Upload a LotteryPost TXT file (tab-delimited) to continue.")
    st.stop()

raw = uploaded.getvalue().decode("utf-8", errors="ignore")
df = parse_history_text(raw)
if df.empty:
    st.error("No usable Pick 3 lines were parsed. Make sure the file is LotteryPost-style tab-delimited TXT.")
    st.stop()

states = sorted(df["state"].unique())
state_sel = st.selectbox("State", states, index=0)
df = df[df["state"] == state_sel].copy()

st.write(f"**Parsed {len(df):,} rows** • State: **{state_sel}**")

# Build streams
df_all = df.sort_values(["date", "draw_order"]).reset_index(drop=True)

if stream_mode == "Per draw time":
    stream_target = (
        df_all[df_all["draw"] == draw].copy().sort_values(["date", "draw_order"]).reset_index(drop=True)
    )
    stream = stream_target  # grid/chart candidates are based on the selected draw-time stream

    transition_mode_final = transition_mode
    transition_summary = None
    if transition_mode == "Auto (choose best)":
        df_eval = df_all.rename(columns={"draw": "draw_time", "num": "result"})
        best, summ = choose_best_transition_mode(
            df_state=df_eval,
            target_draw=draw,
            due_grid_window=due_grid_window,
            chart_lookback=chart_lookback,
            third_digit_sources=third_digit_sources,
            combined_bonus=combined_bonus,
            learn_window=learn_window,
        )
        transition_summary = summ
        transition_mode_final = best

    # Seed choice (what we treat as the previous draw for scoring ±1, Prev-digit, etc.)
    if transition_mode_final == TRANSITION_PREV_OVERALL:
        seed_row = df_all.iloc[-1]
    else:
        seed_row = stream_target.iloc[-1]
else:
    stream_target = df_all
    stream = df_all
    transition_mode_final = TRANSITION_PREV_OVERALL
    transition_summary = None
    seed_row = df_all.iloc[-1]

if transition_summary is not None:
    st.markdown('### Auto transition evaluation')
    st.write('Auto compared seed styles on your history for this draw time (walk-forward).')
    st.dataframe(transition_summary, use_container_width=True)
    st.success(f"Using transition: **{transition_mode_final}**")
nums = stream["num"].tolist()
dates = stream["date"].tolist()

prev_num = str(seed_row["num"]).zfill(3)
prev_date = seed_row["date"]
prev_draw = seed_row.get("draw", "")
# ---------------------------
# Calibration runtime hooks
# ---------------------------
cal_key = f"{state_sel} | {stream_mode} | {draw if stream_mode=='Per draw time' else 'ALL'}"

if clear_cal:
    st.session_state["calibration_store"] = {}
    st.session_state.pop("last_applied_cal_key", None)
    st.success("Saved calibrations cleared.")
    st.rerun()

# Auto-apply a saved calibration for this exact stream (optional)
if auto_apply_cal:
    store = st.session_state.get("calibration_store", {})
    if cal_key in store and st.session_state.get("last_applied_cal_key") != cal_key:
        rec_saved = store[cal_key].get("rec", {})
        if rec_saved:
            st.session_state["use_window"] = True
            st.session_state["drought_window"] = int(rec_saved["window"])
            st.session_state["include_diag"] = bool(rec_saved["include_diagonal"])
            st.session_state["third_pool_mode"] = str(rec_saved["third_pool_mode"])
            st.session_state["last_applied_cal_key"] = cal_key
            st.info("Auto-applied saved calibration for this stream.")
            st.rerun()

# Run a fresh calibration now (button in sidebar)
if calibrate_now:
    if len(nums) < 60:
        st.warning("Not enough history in this stream to calibrate (need ~60+ draws).")
    elif not cal_windows:
        st.warning("Pick at least one grid lookback candidate (X draws).")
    elif not cal_third_pool:
        st.warning("Pick at least one third-digit pool option to test.")
    else:
        windows = tuple(sorted({int(x) for x in cal_windows}))
        diag_opts = (True, False) if cal_try_diag else (bool(include_diag),)
        third_opts = tuple(cal_third_pool)

        df_cal, rec = calibrate_grid_settings(
            nums=nums,
            dates=dates,
            rows_n=rows_n,
            wrap_mod10=wrap_mod10,
            windows=windows,
            diag_options=diag_opts,
            third_pool_options=third_opts,
            pattern_window=pattern_window,
            pair_k=pair_k,
            third_k=third_k,
            pair_digit_prefer=pair_digit_prefer,
            third_digit_prefer=third_digit_prefer,
            recency_include=recency_include,
            pair_strength_window=pair_strength_window,
            max_tests=int(cal_max_tests),
            combined_bonus=float(combined_bonus),
        )

        if df_cal.empty or not rec:
            st.warning("Calibration couldn't find a usable recommendation for this stream.")
        else:
            # Save
            st.session_state["calibration_store"][cal_key] = {
                "rec": rec,
                "table": df_cal,
                "meta": {
                    "state": state_sel,
                    "stream_mode": stream_mode,
                    "draw_time": draw,
                    "n_draws": len(nums),
                    "date_min": _safe_date_str(stream["date"], "min"),
                    "date_max": _safe_date_str(stream["date"], "max"),
                }
            }
            # Apply
            st.session_state["use_window"] = True
            st.session_state["drought_window"] = int(rec["window"])
            st.session_state["include_diag"] = bool(rec["include_diagonal"])
            st.session_state["third_pool_mode"] = str(rec["third_pool_mode"])
            st.session_state["last_applied_cal_key"] = cal_key

            st.success(f"Calibration applied: window={int(rec['window'])}, diag={bool(rec['include_diagonal'])}, third='{rec['third_pool_mode']}'")
            st.rerun()



if len(nums) < 30:
    st.warning(f"Only {len(nums)} draws in the selected stream. Rankings will be unstable with very short history.")

# Use latest draw as the "prev" seed for today's prediction list
# Show saved calibration for this stream (if any)
store = st.session_state.get("calibration_store", {})
if cal_key in store:
    meta = store[cal_key].get("meta", {})
    rec = store[cal_key].get("rec", {})
    df_cal = store[cal_key].get("table", pd.DataFrame())
    st.subheader("Auto-calibration for this stream")
    if rec:
        st.write(
            f"**Recommended (saved):** window={int(rec['window'])}, diagonal={bool(rec['include_diagonal'])}, "
            f"third-pool='{rec['third_pool_mode']}'. "
            f"(history: {meta.get('n_draws','?')} draws, {meta.get('date_min','')} → {meta.get('date_max','')})"
        )
    with st.expander("Calibration table (GRID method, walk-forward)"):
        if isinstance(df_cal, pd.DataFrame) and not df_cal.empty:
            st.dataframe(df_cal, use_container_width=True, hide_index=True)
        else:
            st.write("No table stored.")

prev_digits = set(prev_num)
neigh_digits = neighbors(prev_digits, wrap_mod10=wrap_mod10)

grid, drought = build_due_grid(nums, len(nums), rows=rows_n, window=(drought_window if use_window else None))
strength_fn = digit_pair_strength(grid, include_diagonal=include_diag)
gdigits = grid_digits(grid)

# --- Build Grid list
grid_boxes, third_pool = generate_grid_candidates(grid, strength_fn, prev_digits, neigh_digits, third_pool_mode)
grid_rank = rank_grid_boxes(grid_boxes, drought, prev_digits, neigh_digits, gdigits)

# --- Build Chart list
chart_boxes, chart_info = generate_parity_chart_candidates(
    nums=nums,
    idx_upto_exclusive=len(nums),
    pattern_window=pattern_window,
    pair_k=pair_k,
    third_k=third_k,
    pair_digit_prefer=pair_digit_prefer,
    third_digit_prefer=third_digit_prefer,
    recency_include=recency_include,
    pair_strength_window=pair_strength_window,
)
chart_rank = rank_chart_boxes(chart_boxes, prev_digits, neigh_digits, gdigits)

# --- Combined list
combined_rank = combine_rankings(grid_rank, chart_rank, bonus_if_in_both=combined_bonus)

# Apply adaptive ranking boosts to the **COMBINED** list only (keeps Grid-only and Chart-only rankings unchanged).
if use_adaptive and (not combined_rank.empty):
    try:
        seed_num_int = int(prev_num)
    except Exception:
        seed_num_int = int(combined_rank.iloc[0]["box"]) if len(combined_rank) else 0

    feature_weights = {
        "parity": float(w_parity),
        "hml": float(w_hml),
        "structure": float(w_structure),
        "sum_bucket": float(w_sum_bucket),
        "sum_parity": float(w_sum_parity),
        "root": float(w_root),
        "overlap": float(w_overlap),
        "delta": float(w_delta),
    }

    combined_rank = apply_adaptive_boost_to_combined(
        combined_rank=combined_rank,
        df_stream=stream,
        seed_num=seed_num_int,
        condition_on_seed=bool(condition_on_seed),
        boost_strength=float(boost_strength),
        feature_weights=feature_weights,
        learn_window=learn_window
    )

# Display grid and chart context
st.subheader("Current context (seed + grid + chart pattern)")
c1, c2 = st.columns([1, 2])
with c1:
    st.write(f"**Prev draw used as seed:** {prev_date} → **{prev_num}**")
    st.write(f"Prev digits: {sorted(prev_digits)} • ±1 set: {sorted(neigh_digits)}")

    grid_df = pd.DataFrame(grid, columns=["Hundreds", "Tens", "Ones"])
    st.write(f"**Due Digit Grid (Top {rows_n} overdue digits per position)**")
    st.dataframe(grid_df, hide_index=True)

with c2:
    st.write("**Parity Pair Chart inputs**")
    st.write(f"Dominant pattern (last {pattern_window}): **{chart_info.get('pattern','')}**")
    st.write(f"Pair digits ({pair_digit_prefer}): **{chart_info.get('pair_digits',[])}**")
    st.write(f"Third digits ({third_digit_prefer}): **{chart_info.get('third_digits',[])}**")
    st.write(f"Pairs used: **{chart_info.get('pairs',[])}**")

# Show ranked list
st.subheader("Ranked BOX List")
if strategy_view == "Grid only":
    ranked = grid_rank
    fname_full = f"pick3_{state_sel.lower()}_{'all' if stream_mode!='Per draw time' else draw.lower()}_grid_ranked_boxes.csv"
elif strategy_view == "Chart only":
    ranked = chart_rank
    fname_full = f"pick3_{state_sel.lower()}_{'all' if stream_mode!='Per draw time' else draw.lower()}_chart_ranked_boxes.csv"
else:
    ranked = combined_rank
    fname_full = f"pick3_{state_sel.lower()}_{'all' if stream_mode!='Per draw time' else draw.lower()}_combined_ranked_boxes.csv"

ranked_full = ranked.copy()
ranked_shown = ranked_full

# Optional percentile trim (final cut on the ranked list)
# defaults in case percentile UI is disabled
pct_basis = st.session_state.get("pct_basis", "Full universe (000–999)")
min_n_for_trim = st.session_state.get("min_n_for_trim", 0)
if use_pct_trim:
    ranked_shown = apply_percentile_trim(
        df_ranked=ranked_full,
        mode=pct_mode,
        keep_top_pct=float(keep_top_pct),
        zones=list(zones_keep),
        bin_size=int(zone_bin_size),
        percentile_basis=str(pct_basis),
        min_n_for_trim=int(min_n_for_trim),
    )

# Metrics
m1, m2, m3 = st.columns(3)
m1.metric("Boxes (full)", f"{len(ranked_full):,}")
m2.metric("Boxes (shown)", f"{len(ranked_shown):,}")
if use_pct_trim:
    if pct_mode == "Keep top %":
        m3.metric("Percentile trim", f"Top {keep_top_pct}%")
    else:
        m3.metric("Percentile trim", f"{len(zones_keep)} zones")
else:
    m3.metric("Percentile trim", "OFF")

# Strategy context metrics
cA, cB, cC = st.columns(3)
if strategy_view == "Grid only":
    cA.metric("Grid digits", f"{len(gdigits):,}")
    cB.metric("Third pool size", f"{len(third_pool):,}")
    cC.metric("Diagonal pairs", "ON" if include_diag else "OFF")
elif strategy_view == "Chart only":
    cA.metric("Pairs in play", f"{len(chart_info.get('pairs',[])):,}")
    cB.metric("Pair-strength window", f"{pair_strength_window:,}")
    cC.metric("Pattern window", f"{pattern_window:,}")
else:
    cA.metric("In BOTH", f"{int((ranked_full['in_grid'] & ranked_full['in_chart']).sum()) if 'in_grid' in ranked_full.columns else 0:,}")
    cB.metric("Combined bonus", f"{combined_bonus:g}")
    cC.metric("Adaptive boosts", "ON" if use_adaptive else "OFF")

# Display
st.dataframe(ranked_shown, use_container_width=True, hide_index=True)

# Downloads
full_bytes = ranked_full.to_csv(index=False).encode("utf-8")
st.download_button("Download FULL ranked list (CSV)", data=full_bytes, file_name=fname_full, mime="text/csv")

if use_pct_trim and (len(ranked_shown) != len(ranked_full)):
    fname_trim = fname_full.replace(".csv", f"_TRIM_{pct_mode.replace(' ','_').lower()}.csv")
    trim_bytes = ranked_shown.to_csv(index=False).encode("utf-8")
    st.download_button("Download TRIMMED ranked list (CSV)", data=trim_bytes, file_name=fname_trim, mime="text/csv")

if use_pct_trim and pct_mode == "Select zones":
    st.caption("Tip: run the backtest below (or click 'Auto-pick zones') to see which percentile zones historically contained winners for your current settings.")

# Backtest + percentile-zone analysis
bt_detail = None
bt_summary = None

# Run backtest when requested, or when needed to auto-pick percentile zones
need_bt_for_zones = bool(use_pct_trim and pct_mode == "Select zones" and zone_auto_pick_btn)

if run_bt or need_bt_for_zones:
    if run_bt:
        st.subheader("Method comparison backtest (walk-forward, no look-ahead)")
    bt_detail, bt_summary = run_backtest(
        nums=nums,
        dates=dates,
        rows_n=rows_n,
        include_diag=include_diag,
        third_pool_mode=third_pool_mode,
        wrap_mod10=wrap_mod10,
        use_window=use_window,
        drought_window=drought_window,
        pattern_window=pattern_window,
        pair_k=pair_k,
        third_k=third_k,
        pair_digit_prefer=pair_digit_prefer,
        third_digit_prefer=third_digit_prefer,
        recency_include=recency_include,
        pair_strength_window=pair_strength_window,
        max_tests=max_tests,
        combined_bonus=combined_bonus,
    )

    if bt_summary is None or bt_summary.empty:
        if run_bt:
            st.warning("Not enough history to run the backtest with the current settings.")
    else:
        if run_bt:
            st.write("**Summary metrics**")
            st.dataframe(bt_summary, use_container_width=True, hide_index=True)
            st.write("Interpretation tips:")
            st.write("- **hit_rate**: how often the winner’s BOX was somewhere in the method’s full list.")
            st.write("- **avg_rank_pct**: average winner rank as a percentile of list size (lower is better).")
            st.write("- **top20_rate/top50_rate**: how often the winner appeared in the top 20 / top 50 boxes.")
            with st.expander("Backtest detail (per draw)"):
                st.dataframe(bt_detail, use_container_width=True, hide_index=True)

# Percentile-zone analysis (uses backtest winner-rank percentiles)
if use_pct_trim and pct_mode == "Select zones" and bt_detail is not None and not bt_detail.empty:
    prefix = "combined" if strategy_view == "Combined (union)" else ("grid" if strategy_view == "Grid only" else "chart")
    zt = build_percentile_zone_table(bt_detail, prefix=prefix, bin_size=int(zone_bin_size))

    if not zt.empty:
        with st.expander("Percentile zones (where past winners ranked)"):
            st.write("This table shows which rank-percentile zones contained winners during walk-forward backtests for your *current settings*.")
            st.dataframe(zt, use_container_width=True, hide_index=True)

        if zone_auto_pick_btn:
            rec_zones = recommend_zones_for_target(zt, target_coverage=float(zone_target_cov))
            if rec_zones:
                st.session_state["zones_keep"] = rec_zones
                st.success(f"Auto-picked {len(rec_zones)} zones to reach ~{int(zone_target_cov*100)}% historical winner coverage.")
                st.rerun()
            else:
                st.warning("Could not auto-pick zones (not enough usable backtest hits).")


# ---------------------------
# 7) Filter Lab (optional)
# ---------------------------
st.subheader("Filter Lab (optional) — import DC-5 style filters, adapt to Pick 3, and test")

with st.expander("Why this exists (quick)"):
    st.write(
        "Percentile trims are *rank-based* — they can drop a real winner if that winner scores low under the current scoring model. "
        "A filter lab lets you add *rule-based* cuts (hot/cold/due/mirror/shared-digits/sum/spread/etc.) and measure two things: "
        "(1) how much it reduces the pool and (2) how often it keeps the real next-draw winner in walk-forward tests."
    )

# Helper: safe-ish eval for filter expressions
_ALLOWED_FUNCS = {
    'any': any,
    'all': all,
    'sum': sum,
    'len': len,
    'set': set,
    'max': max,
    'min': min,
    'abs': abs,
}

def _to_digits3(box_str: str):
    s = str(box_str).zfill(3)
    return [int(s[0]), int(s[1]), int(s[2])]

_MIRROR = {0:5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}
_VTRAC = {0:0, 5:0, 1:1, 6:1, 2:2, 7:2, 3:3, 8:3, 4:4, 9:4}


def _compute_hot_cold_digits(df_stream: pd.DataFrame, window: int = 50, hot_k: int = 3, cold_k: int = 3):
    """Simple, transparent hot/cold definition for Pick 3:
    - hot: most frequent digits across last `window` draws (ties included)
    - cold: least frequent digits across last `window` draws (ties included)
    """
    if df_stream is None or df_stream.empty:
        return [], []

    w = min(int(window), len(df_stream))
    tail = df_stream.tail(w)
    digits = []
    for n in tail['num'].astype(str).tolist():
        n = str(n).zfill(3)
        digits.extend([int(n[0]), int(n[1]), int(n[2])])

    if not digits:
        return [], []

    from collections import Counter
    c = Counter(digits)

    # sort by freq then digit
    items = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
    hot_cut = items[:max(1, int(hot_k))]
    hot_digits = sorted({d for d,_ in hot_cut})

    items2 = sorted(c.items(), key=lambda kv: (kv[1], kv[0]))
    cold_cut = items2[:max(1, int(cold_k))]
    cold_digits = sorted({d for d,_ in cold_cut})

    return hot_digits, cold_digits


def _adapt_dc5_expr_to_pick3(expr: str):
    """Best-effort adapter: removes DC-5-only guards and rescales obvious thresholds.

    IMPORTANT: This is intentionally conservative — it does *not* try to be clever.
    It mainly:
      - strips `combo_structure == 5`
      - replaces len(set(combo_digits))==5 with ==3
      - rescales sum thresholds from 0–45 to 0–27 (linear)
      - maps shared-digit thresholds (>=4 -> >=2, >=5 -> >=3)

    Anything obviously impossible for Pick 3 (sum==37, sum>40, etc.) is flagged as incompatible.
    """
    if expr is None:
        return "", False
    e = str(expr).strip()
    if e == "" or e.lower() == "nan":
        return "", False

    # quick incompatibility check for impossible constants
    # (not perfect, but catches the worst offenders)
    for m in re.findall(r"combo_sum\s*==\s*(\d+)", e):
        if int(m) > 27:
            return e, False
    for m in re.findall(r"combo_sum\s*>\s*(\d+)", e):
        if int(m) > 27:
            # we will rescale later, but if it's *way* out of range like >40, handle in rescale
            pass

    # remove DC-5 structure guard
    e = re.sub(r"\s*and\s*combo_structure\s*==\s*5\s*", " ", e)
    e = re.sub(r"\s*combo_structure\s*==\s*5\s*and\s*", " ", e)
    e = re.sub(r"combo_structure\s*==\s*5", "True", e)

    # Replace '5 unique' logic with Pick3 'single'
    e = e.replace("len(set(combo_digits))==5", "len(set(combo_digits))==3")
    e = e.replace("len(set(combo_digits)) == 5", "len(set(combo_digits)) == 3")

    # V-trac group count ==5 -> ==3 for pick3
    e = e.replace("== 5", "== 3") if "combo_vtracs" in e and "== 5" in e else e

    # Shared-digit thresholds map (DC5 >=4/5 => Pick3 >=2/3)
    e = re.sub(r"(sum\([^\)]*d\s+in\s+combo_digits[^\)]*\)\s*>=)\s*5", r"\1 3", e)
    e = re.sub(r"(sum\([^\)]*d\s+in\s+combo_digits[^\)]*\)\s*>=)\s*4", r"\1 2", e)

    # Rescale sum thresholds 0–45 -> 0–27 for simple numeric comparisons
    def _scale(n: int) -> int:
        return int(round(n * 27 / 45))

    def _sub_cmp(match):
        op = match.group(1)
        n = int(match.group(2))
        return f"combo_sum {op} {_scale(n)}"

    e = re.sub(r"combo_sum\s*(>=|<=|>|<)\s*(\d+)", _sub_cmp, e)

    # Another incompatibility check after scaling for exact equals above range
    for m in re.findall(r"combo_sum\s*==\s*(\d+)", e):
        if int(m) > 27:
            return e, False

    return e, True


def _safe_eval(expr: str, ctx: dict) -> bool:
    try:
        return bool(eval(expr, {"__builtins__": {} , **_ALLOWED_FUNCS}, ctx))
    except Exception:
        return False


# UI: upload filter file
flt_file = st.file_uploader("Upload filter file (CSV/TXT exported from your tester)", type=["csv", "txt"], key="flt_file")
flt_path_default = None
try:
    # if running locally with this repo, you can place a default file next to the app
    # Streamlit Cloud won't have /mnt/data; this is just a convenience for local dev.
    flt_path_default = "try pk3 lotto batch 10.txt"
except Exception:
    flt_path_default = None

use_adapter = st.checkbox("Auto-adapt DC-5 expressions to Pick 3 (recommended)", value=True, key="use_adapter")

hc_col1, hc_col2, hc_col3 = st.columns(3)
with hc_col1:
    hc_window = st.number_input("Hot/Cold window (last N draws)", min_value=10, max_value=500, value=int(st.session_state.get("hc_window", 50)), step=5, key="hc_window")
with hc_col2:
    hot_k = st.number_input("Hot K", min_value=1, max_value=9, value=int(st.session_state.get("hot_k", 3)), step=1, key="hot_k")
with hc_col3:
    cold_k = st.number_input("Cold K", min_value=1, max_value=9, value=int(st.session_state.get("cold_k", 3)), step=1, key="cold_k")

tracked_box = st.text_input("Track a specific BOX (optional) — e.g., 026", value=str(st.session_state.get("tracked_box", "")), key="tracked_box")

filters_df = None
if flt_file is not None:
    try:
        filters_df = pd.read_csv(flt_file)
    except Exception:
        filters_df = None
elif flt_path_default and Path(flt_path_default).exists():
    try:
        filters_df = pd.read_csv(flt_path_default)
    except Exception:
        filters_df = None

if filters_df is None or filters_df.empty:
    st.info("Upload your filter file above to enable this section.")
else:
    # normalize columns
    for col in ["id", "name", "enabled", "applicable_if", "expression"]:
        if col not in filters_df.columns:
            st.error(f"Filter file is missing required column: {col}")
            filters_df = None
            break

if filters_df is not None and not filters_df.empty:
    # clean
    dfF = filters_df[["id", "name", "enabled", "applicable_if", "expression"]].copy()
    dfF["enabled"] = dfF["enabled"].astype(str).str.lower().isin(["true", "1", "yes", "y"])

    # Adapt expressions
    expr_pick3 = []
    app_pick3 = []
    compat = []
    for _, r in dfF.iterrows():
        app = str(r.get("applicable_if", "True")).strip()
        expr = str(r.get("expression", "True")).strip()

        if use_adapter:
            expr2, ok2 = _adapt_dc5_expr_to_pick3(expr)
            app2, okA = _adapt_dc5_expr_to_pick3(app) if app not in ["", "nan", "None"] else ("True", True)
            ok = bool(ok2 and okA)
        else:
            expr2, app2, ok = expr, app, True

        expr_pick3.append(expr2)
        app_pick3.append(app2)
        compat.append(ok)

    dfF["expr_pick3"] = expr_pick3
    dfF["app_pick3"] = app_pick3
    dfF["compatible_pick3"] = compat

    # Show filter list
    with st.expander("Filter list (adapted)"):
        st.dataframe(dfF[["id", "name", "enabled", "compatible_pick3", "app_pick3", "expr_pick3"]], use_container_width=True, hide_index=True)

    usable = dfF[dfF["enabled"] & dfF["compatible_pick3"]].copy()
    if usable.empty:
        st.warning("No enabled + compatible filters found after adaptation.")
    else:
        # pick filters to apply
        ids = usable["id"].astype(str).tolist()
        default_ids = ids[: min(5, len(ids))]
        selected_ids = st.multiselect(
            "Select filters to APPLY (in this exact order)",
            options=ids,
            default=st.session_state.get("selected_filter_ids", default_ids),
            key="selected_filter_ids",
        )

        # Build evaluation context pieces from current stream + seed
        hot_digits, cold_digits = _compute_hot_cold_digits(stream, window=int(hc_window), hot_k=int(hot_k), cold_k=int(cold_k))
        st.caption(f"Computed Hot digits: {hot_digits} • Cold digits: {cold_digits} (window={int(hc_window)})")

        seed_digits = sorted(prev_digits)
        seed_sum = sum(seed_digits)
        seed_counts = {d: seed_digits.count(d) for d in range(10)}
        seed_vtracs = { _VTRAC[int(d)] for d in seed_digits }

        def _apply_filters_to_ranked(df_ranked_in: pd.DataFrame):
            df_work = df_ranked_in.copy()
            kept_log = []

            # track box check
            tbox = str(tracked_box).strip()
            if tbox:
                tbox = tbox.zfill(3)

            for fid in selected_ids:
                row = usable[usable['id'].astype(str)==str(fid)].iloc[0]
                app_expr = str(row['app_pick3'])
                expr = str(row['expr_pick3'])

                before_n = len(df_work)
                if before_n == 0:
                    kept_log.append({"id": fid, "name": row['name'], "before": 0, "after": 0, "eliminated": 0, "tracked_kept": False})
                    continue

                keep_mask = []
                for bx in df_work['box'].astype(str).tolist():
                    combo_digits = _to_digits3(bx)
                    combo_sum = sum(combo_digits)
                    combo_vtracs = { _VTRAC[int(d)] for d in combo_digits }
                    combo_structure = 3  # kept for legacy expressions; always 3 digits
                    ctx = {
                        'combo_digits': combo_digits,
                        'combo_sum': combo_sum,
                        'combo_vtracs': combo_vtracs,
                        'seed_digits': seed_digits,
                        'seed_sum': seed_sum,
                        'seed_value': seed_sum,
                        'seed_counts': seed_counts,
                        'seed_vtracs': seed_vtracs,
                        'mirror': _MIRROR,
                        'hot_digits': hot_digits,
                        'cold_digits': cold_digits,
                        'combo_structure': combo_structure,
                    }

                    applicable = _safe_eval(app_expr, ctx) if app_expr else True
                    if not applicable:
                        keep_mask.append(True)
                    else:
                        eliminate = _safe_eval(expr, ctx) if expr else False
                        keep_mask.append(not eliminate)

                df_work = df_work.loc[keep_mask].reset_index(drop=True)
                after_n = len(df_work)

                tracked_kept = False
                if tbox:
                    tracked_kept = (tbox in df_work['box'].astype(str).str.zfill(3).tolist())

                kept_log.append({
                    "id": fid,
                    "name": row['name'],
                    "before": before_n,
                    "after": after_n,
                    "eliminated": before_n - after_n,
                    "tracked_kept": tracked_kept,
                })

            return df_work, pd.DataFrame(kept_log)

        # Apply to current ranked list (post-percentile trim)
        st.write("### Apply filters to the CURRENT ranked list")
        apply_btn = st.button("Apply selected filters to current list", key="apply_filters_now")

        if apply_btn:
            filtered_df, log_df = _apply_filters_to_ranked(ranked_shown)

            st.write("**Filter-by-filter impact**")
            st.dataframe(log_df, use_container_width=True, hide_index=True)

            st.write(f"**Remaining after all selected filters:** {len(filtered_df):,} boxes")
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)

            st.download_button(
                "Download FILTERED ranked list (CSV)",
                data=filtered_df.to_csv(index=False).encode('utf-8'),
                file_name="pick3_filtered_ranked_list.csv",
                mime="text/csv",
            )

        # Walk-forward test for selected filters
        st.write("### Walk-forward efficiency test (keep-winner vs reduction)")
        test_n = st.slider("Test last N transitions", min_value=25, max_value=500, value=int(st.session_state.get('filter_bt_n', 150)), step=25, key='filter_bt_n')
        run_filter_bt = st.button("Run filter backtest", key="run_filter_bt")

        if run_filter_bt:
            if len(selected_ids) == 0:
                st.warning("Select at least one filter first.")
            else:
                # Use the same transition logic as your app: each step predicts next draw from previous draw
                # We'll reuse run_backtest's underlying generation logic by doing a lightweight loop.
                # (This is a simplified backtest focused on filter impact, not method comparison.)

                # Prepare series
                nums_s = nums.copy()
                dates_s = dates.copy()
                if len(nums_s) < 5:
                    st.warning("Not enough history for filter backtest.")
                else:
                    # limit to last N transitions
                    nT = min(int(test_n), len(nums_s) - 1)
                    start_idx = max(1, len(nums_s) - nT)

                    hits = 0
                    total = 0
                    sizes_before = []
                    sizes_after = []

                    detail_rows = []

                    for i in range(start_idx, len(nums_s)):
                        seed = str(nums_s[i-1]).zfill(3)
                        actual = str(nums_s[i]).zfill(3)
                        actual_box = "".join(sorted(actual))  # box-style compare

                        # Build per-step stream slice up to seed time (no look-ahead)
                        df_hist = stream.iloc[:i].copy() if (stream is not None and len(stream) >= i) else stream.copy()

                        # Recompute grid + chart for this step using existing helpers
                        seed_digits_step = [int(seed[0]), int(seed[1]), int(seed[2])]
                        neigh_step = set((d+1) % 10 for d in seed_digits_step) | set((d-1) % 10 for d in seed_digits_step)

                        grid_step = build_due_digit_grid(df_hist, rows_n=rows_n, use_window=use_window, drought_window=drought_window)
                        gdigits_step = set(int(x) for x in grid_step.flatten() if str(x).strip() != '')

                        chart_step = build_parity_pair_chart(df_hist, window=pattern_window, pair_k=pair_k, third_k=third_k,
                                                           pair_digit_prefer=pair_digit_prefer, third_digit_prefer=third_digit_prefer,
                                                           recency_include=recency_include, pair_strength_window=pair_strength_window)

                        # Candidate boxes (grid + chart) per current configuration
                        # For filter-testing, we use the Combined union list as the baseline candidate set.
                        third_pool_step = build_third_digit_pool(df_hist, seed_digits_step, neigh_step, gdigits_step, mode=third_pool_mode,
                                                                third_k=third_k, wrap_mod10=wrap_mod10)
                        grid_boxes_step = generate_grid_boxes(gdigits_step, third_pool_step, include_diag=include_diag)
                        chart_boxes_step = generate_chart_boxes(chart_step, seed_digits_step, neigh_step, gdigits_step)

                        # Rank and combine
                        gr = rank_grid_boxes(grid_boxes_step, seed_digits_step, neigh_step, gdigits_step)
                        cr = rank_chart_boxes(chart_boxes_step, seed_digits_step, neigh_step, gdigits_step)
                        comb = combine_rankings(gr, cr, bonus_if_in_both=combined_bonus)

                        # Optional adaptive boosts (match app)
                        if use_adaptive and (not comb.empty):
                            feature_weights = {
                                'parity': float(w_parity),
                                'hml': float(w_hml),
                                'structure': float(w_structure),
                                'sum_bucket': float(w_sum_bucket),
                                'sum_parity': float(w_sum_parity),
                                'root': float(w_root),
                                'overlap': float(w_overlap),
                                'delta': float(w_delta),
                            }
                            try:
                                seed_num_int = int(seed)
                            except Exception:
                                seed_num_int = 0
                            comb = apply_adaptive_boost_to_combined(
                                combined_rank=comb,
                                df_stream=df_hist,
                                seed_num=seed_num_int,
                                condition_on_seed=bool(condition_on_seed),
                                boost_strength=float(boost_strength),
                                feature_weights=feature_weights,
                                learn_window=learn_window,
                            )

                        base_list = comb.copy()

                        # Apply percentile trim the same way as current settings
                        if use_pct_trim:
                            base_list = apply_percentile_trim(
                                df_ranked=base_list,
                                mode=pct_mode,
                                keep_top_pct=float(keep_top_pct),
                                zones=list(zones_keep),
                                bin_size=int(zone_bin_size),
                                percentile_basis=str(pct_basis),
                                min_n_for_trim=int(min_n_for_trim),
                            )

                        sizes_before.append(len(base_list))

                        # Recompute hot/cold for the step
                        hot_step, cold_step = _compute_hot_cold_digits(df_hist, window=int(hc_window), hot_k=int(hot_k), cold_k=int(cold_k))

                        # Apply selected filters
                        # (We rebuild the small context globals for each step)
                        seed_sum_step = sum(seed_digits_step)
                        seed_counts_step = {d: seed_digits_step.count(d) for d in range(10)}
                        seed_vtracs_step = { _VTRAC[int(d)] for d in seed_digits_step }

                        df_work = base_list.copy()
                        for fid in selected_ids:
                            row = usable[usable['id'].astype(str)==str(fid)].iloc[0]
                            app_expr = str(row['app_pick3'])
                            expr = str(row['expr_pick3'])

                            keep_mask = []
                            for bx in df_work['box'].astype(str).tolist():
                                combo_digits = _to_digits3(bx)
                                combo_sum = sum(combo_digits)
                                combo_vtracs = { _VTRAC[int(d)] for d in combo_digits }
                                combo_structure = 3
                                ctx = {
                                    'combo_digits': combo_digits,
                                    'combo_sum': combo_sum,
                                    'combo_vtracs': combo_vtracs,
                                    'seed_digits': seed_digits_step,
                                    'seed_sum': seed_sum_step,
                                    'seed_value': seed_sum_step,
                                    'seed_counts': seed_counts_step,
                                    'seed_vtracs': seed_vtracs_step,
                                    'mirror': _MIRROR,
                                    'hot_digits': hot_step,
                                    'cold_digits': cold_step,
                                    'combo_structure': combo_structure,
                                }

                                applicable = _safe_eval(app_expr, ctx) if app_expr else True
                                if not applicable:
                                    keep_mask.append(True)
                                else:
                                    eliminate = _safe_eval(expr, ctx) if expr else False
                                    keep_mask.append(not eliminate)

                            df_work = df_work.loc[keep_mask].reset_index(drop=True)

                        sizes_after.append(len(df_work))

                        # Determine hit by BOX (orderless)
                        pred_boxes = {"".join(sorted(str(b).zfill(3))) for b in df_work['box'].astype(str).tolist()}
                        hit = (actual_box in pred_boxes)
                        hits += int(hit)
                        total += 1

                        detail_rows.append({
                            'seed': seed,
                            'actual': actual,
                            'hit': hit,
                            'before': len(base_list),
                            'after': len(df_work),
                        })

                    if total == 0:
                        st.warning("No transitions to test (check history length / settings).")
                    else:
                        import numpy as np
                        hit_rate = hits / total
                        avg_before = float(np.mean(sizes_before)) if sizes_before else 0.0
                        avg_after = float(np.mean(sizes_after)) if sizes_after else 0.0
                        med_before = float(np.median(sizes_before)) if sizes_before else 0.0
                        med_after = float(np.median(sizes_after)) if sizes_after else 0.0

                        st.write("**Results**")
                        st.write(f"Winner BOX kept: **{hits}/{total}**  (hit rate **{hit_rate:.1%}**)\n\nAverage pool: **{avg_before:.1f} → {avg_after:.1f}** (median **{med_before:.0f} → {med_after:.0f}**) ")

                        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
