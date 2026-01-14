# pick3_touch_grid_app_v8.py
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
from collections import Counter, defaultdict

import pandas as pd
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
    draw = st.selectbox("Draw time (if Per draw time)", ["Morning", "Day", "Evening", "Night"], index=0, key="draw_time")

    st.header("3) Grid strategy settings")
    rows_n = st.slider("Grid rows (top overdue digits per position)", min_value=3, max_value=6, value=st.session_state.get("rows_n", 4), step=1, key="rows_n")
    include_diag = st.checkbox("Count diagonal touches (catacorner)", value=st.session_state.get("include_diag", True), key="include_diag")

    st.subheader("Third-digit pool (Grid strategy)")
    third_pool_mode = st.selectbox(
        "Third digit can come from:",
        ["Grid + Prev + ±1", "Grid + Prev", "Grid only", "All digits 0–9"],
        index=0,
        key="third_pool_mode"
    )
    wrap_mod10 = st.checkbox("±1 uses wrap-around (0↔9)", value=st.session_state.get("wrap_mod10", False), key="wrap_mod10")

    st.subheader("Optional: drought window (replicates the 'last N draws' grid)")
    use_window = st.checkbox("Limit drought calc to last X draws", value=st.session_state.get("use_window", False), key="use_window")
    drought_window = st.slider("X draws", min_value=6, max_value=200, value=int(st.session_state.get("drought_window", 30)), step=1, disabled=not use_window, key="drought_window")

    st.header("4) Parity Pair Chart settings")
    pattern_window = st.slider("Pattern window (N draws)", min_value=5, max_value=60, value=int(st.session_state.get("pattern_window", 10)), step=1, key="pattern_window")
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

# Build stream
if stream_mode == "Per draw time":
    stream = df[df["draw"] == draw].sort_values(["date", "draw_order"]).reset_index(drop=True)
else:
    stream = df.sort_values(["date", "draw_order"]).reset_index(drop=True)

nums = stream["num"].tolist()
dates = stream["date"].tolist()

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
                    "date_min": str(stream["date"].min().date()) if not stream.empty else "",
                    "date_max": str(stream["date"].max().date()) if not stream.empty else "",
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

prev_num = nums[-1]
prev_date = dates[-1]
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
st.subheader("FULL Ranked BOX List")
if strategy_view == "Grid only":
    ranked = grid_rank
    fname = f"pick3_{state_sel.lower()}_{'all' if stream_mode!='Per draw time' else draw.lower()}_grid_ranked_boxes.csv"
elif strategy_view == "Chart only":
    ranked = chart_rank
    fname = f"pick3_{state_sel.lower()}_{'all' if stream_mode!='Per draw time' else draw.lower()}_chart_ranked_boxes.csv"
else:
    ranked = combined_rank
    fname = f"pick3_{state_sel.lower()}_{'all' if stream_mode!='Per draw time' else draw.lower()}_combined_ranked_boxes.csv"

cA, cB, cC = st.columns(3)
cA.metric("Boxes in list", f"{len(ranked):,}")
if strategy_view == "Grid only":
    cB.metric("Grid digits", f"{len(gdigits):,}")
    cC.metric("Third pool size", f"{len(third_pool):,}")
elif strategy_view == "Chart only":
    cB.metric("Pairs in play", f"{len(chart_info.get('pairs',[])):,}")
    cC.metric("Pair-strength window", f"{pair_strength_window:,}")
else:
    cB.metric("In BOTH", f"{int((ranked['in_grid'] & ranked['in_chart']).sum()) if 'in_grid' in ranked.columns else 0:,}")
    cC.metric("Combined bonus", f"{combined_bonus:g}")

st.dataframe(ranked, use_container_width=True, hide_index=True)

csv_bytes = ranked.to_csv(index=False).encode("utf-8")
st.download_button("Download full ranked list (CSV)", data=csv_bytes, file_name=fname, mime="text/csv")

# Backtest comparison
if run_bt:
    st.subheader("Method comparison backtest (walk-forward, no look-ahead)")
    detail, summary = run_backtest(
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
    if summary.empty:
        st.warning("Not enough history to run the backtest with the current settings.")
    else:
        st.write("**Summary metrics**")
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.write("Interpretation tips:")
        st.write("- **hit_rate**: how often the winner’s BOX was somewhere in the method’s full list.")
        st.write("- **avg_rank_pct**: average winner rank as a percentile of list size (lower is better).")
        st.write("- **top20_rate/top50_rate**: how often the winner appeared in the top 20 / top 50 boxes.")
        with st.expander("Backtest detail (per draw)"):
            st.dataframe(detail, use_container_width=True, hide_index=True)
