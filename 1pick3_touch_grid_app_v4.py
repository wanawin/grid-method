import re
import math
import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import pandas as pd
import streamlit as st

DIGITS = list(range(10))

DRAW_TIME_ALIASES = {
    "morning": "Morning",
    "morn": "Morning",
    "am": "Morning",
    "day": "Day",
    "noon": "Day",
    "midday": "Day",
    "evening": "Evening",
    "eve": "Evening",
    "night": "Night",
    "pm": "Night",
}

DEFAULT_WEIGHTS = {
    "pos_overdue": 1.0,     # per-position overdue distance
    "touch": 3.0,           # per touching-pair edge strength
    "row_bonus": 6.0,       # if candidate equals a direct grid row combo
    "row_rank_decay": 1.2,  # row 0 gets row_bonus, row 1 gets row_bonus/decay, ...
    "doubles": 4.0,         # doubled digit is "supported" by doubles model
    "carryover": 2.0,       # digit appears in previous winner
    "neighbor": 1.5,        # digit is +/- 1 from a previous winner digit
}

@dataclass
class Grid:
    rows: int
    cols: int
    cells: List[List[int]]  # shape rows x cols

    def digits_set(self) -> Set[int]:
        return set(d for r in self.cells for d in r)

def _normalize_draw_time(x: str) -> str:
    if not isinstance(x, str):
        return ""
    k = x.strip().lower()
    return DRAW_TIME_ALIASES.get(k, x.strip())

def _extract_pick3_numbers_from_text(text: str) -> List[str]:
    """Extract Pick-3 results from arbitrary pasted text.

    Supports both:
      - '123' style results
      - '1-2-3' / '1–2–3' / '1 — 2 — 3' style results
    """
    results: List[str] = []

    # 1) Hyphen/ndash/mdash separated digits (common in LotteryPost exports)
    for m in re.finditer(r"\b(\d)\s*[-–—]\s*(\d)\s*[-–—]\s*(\d)\b", text):
        results.append(f"{m.group(1)}{m.group(2)}{m.group(3)}")

    # 2) Plain 3-digit results
    for m in re.finditer(r"\b\d{3}\b", text):
        results.append(m.group(0))

    # Keep order but de-duplicate exact repeats only if they are adjacent copies
    cleaned: List[str] = []
    for s in results:
        s = s.zfill(3)
        if not cleaned or cleaned[-1] != s:
            cleaned.append(s)
    return cleaned


def _ensure_3digits(s: str) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None

    # Already 1-3 digits
    if re.fullmatch(r"\d{1,3}", s):
        return s.zfill(3)

    # Hyphen/space separated digits like 7-7-4
    m = re.search(r"\b(\d)\s*[-–—\s]\s*(\d)\s*[-–—\s]\s*(\d)\b", s)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"

    return None
    if re.fullmatch(r"\d{1,3}", s):
        return s.zfill(3)
    return None

def _parse_simple_numbers(text: str, reverse_order: bool) -> List[str]:
    nums = _extract_pick3_numbers_from_text(text)
    if reverse_order:
        nums = list(reversed(nums))
    return nums

def _guess_result_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        lc = str(col).strip().lower()
        if lc in {"result", "winning", "winning_number", "winning numbers", "numbers"}:
            return col
    best_col = None
    best_hits = 0
    for col in df.columns:
        ser = df[col].astype(str)
        hits = ser.str.fullmatch(r"\s*\d{1,3}\s*").fillna(False).sum()
        if int(hits) > best_hits:
            best_hits = int(hits)
            best_col = col
    return best_col if best_hits > 0 else None

def _parse_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    date_col = None
    for col in out.columns:
        if str(col).strip().lower() in {"date", "draw_date"}:
            date_col = col
            break

    draw_col = None
    for col in out.columns:
        if str(col).strip().lower() in {"draw", "draw_time", "time", "drawtime"}:
            draw_col = col
            break

    res_col = _guess_result_column(out)
    if res_col is None:
        txt = out.to_csv(index=False)
        nums = _extract_pick3_numbers_from_text(txt)
        return pd.DataFrame({"result": nums})

    out["result"] = out[res_col].apply(_ensure_3digits)
    out = out.dropna(subset=["result"]).copy()

    if draw_col is not None:
        out["draw_time"] = out[draw_col].astype(str).apply(_normalize_draw_time)
    else:
        out["draw_time"] = ""

    if date_col is not None:
        out["date"] = pd.to_datetime(out[date_col], errors="coerce")
    else:
        out["date"] = pd.NaT

    keep = ["date", "draw_time", "result"]
    for k in keep:
        if k not in out.columns:
            out[k] = pd.NaT if k == "date" else ""
    out = out[keep].copy()

    if out["date"].notna().any():
        out = out.sort_values(["date", "draw_time", "result"], kind="stable")
    return out.reset_index(drop=True)

def _parse_lotterypost_export_text(text: str) -> pd.DataFrame:
    """Parse LotteryPost-style tab-separated exports.

    Expected line shape (tabs):
      <date> \t <state> \t <game + draw time> \t <result + extras>

    Example:
      Sat, Jan 10, 2026\tTexas\tPick 3 Morning\t7-7-4, Fireball: 3
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue

        date_str = parts[0].strip()
        draw_str = parts[2].strip()
        result_str = parts[3].strip()

        dt = pd.to_datetime(date_str, format="%a, %b %d, %Y", errors="coerce")
        if pd.isna(dt):
            # Fallback parse
            dt = pd.to_datetime(date_str, errors="coerce")

        m = re.search(r"Pick\s*3\s*(Morning|Day|Evening|Night)", draw_str, flags=re.I)
        draw_time = m.group(1).title() if m else ""

        # result: prefer hyphen-separated 1-2-3
        m2 = re.search(r"\b(\d)\s*[-–—]\s*(\d)\s*[-–—]\s*(\d)\b", result_str)
        if m2:
            result = f"{m2.group(1)}{m2.group(2)}{m2.group(3)}"
        else:
            m3 = re.search(r"\b(\d{3})\b", result_str)
            result = m3.group(1) if m3 else None

        result = _ensure_3digits(result) if result is not None else None
        if result is None:
            continue

        rows.append((dt, draw_time, result))

    df_out = pd.DataFrame(rows, columns=["date", "draw_time", "result"]).dropna(subset=["date"])
    order = {"Morning": 0, "Day": 1, "Evening": 2, "Night": 3}
    df_out["_order"] = df_out["draw_time"].map(order).fillna(99).astype(int)
    df_out = df_out.sort_values(["date", "_order"]).drop(columns=["_order"]).reset_index(drop=True)
    return df_out


def _digits_of(num3: str) -> Tuple[int, int, int]:
    return int(num3[0]), int(num3[1]), int(num3[2])


def _structure_type(num3: str) -> str:
    a, b, c = _digits_of(num3)
    if a == b == c:
        return "Triple"
    if a == b or b == c or a == c:
        return "Double"
    return "Single"

def _compute_structure_priors(history: List[str]) -> Dict[str, float]:
    """Return empirical structure probabilities from a list of 3-digit results."""
    if not history:
        return {"Single": 1/3, "Double": 1/3, "Triple": 1/3}
    counts = {"Single": 0, "Double": 0, "Triple": 0}
    for s in history:
        t = _structure_type(s)
        counts[t] += 1
    total = sum(counts.values()) or 1
    return {k: counts[k] / total for k in counts}


def _droughts_and_current(structures: List[str], target: str) -> Tuple[List[int], int]:
    """Return (completed drought lengths, current drought length) for a target structure.

    - A 'drought' is the number of draws between occurrences of the target.
      Example: if targets appear at positions ..., the drought lengths are the counts of non-target draws between them.
    - 'current' is the number of most-recent draws since the last target (0 if the latest draw is the target).
    """
    droughts: List[int] = []
    current = 0
    for s in structures:
        if s == target:
            droughts.append(current)
            current = 0
        else:
            current += 1
    return droughts, current

def _due_metrics_from_history(history: List[str], target: str) -> Optional[Dict[str, float]]:
    """Compute 'due' metrics for a target structure from a list of 3-digit results.

    Returns None if there aren't enough target events to estimate metrics.
    """
    if not history:
        return None
    structs = [_structure_type(x) for x in history]
    droughts, current = _droughts_and_current(structs, target=target)
    if len(droughts) < 5:
        return None

    s = pd.Series(droughts, dtype="float")
    p90 = float(s.quantile(0.90, interpolation="linear"))
    p95 = float(s.quantile(0.95, interpolation="linear"))
    p99 = float(s.quantile(0.99, interpolation="linear"))
    mx = float(s.max())

    # Percentile: where the current drought sits among historical droughts
    percentile = float((s <= current).mean())  # higher => longer/more-rare drought
    exceedance = float((s >= current).mean())  # fraction of historical droughts at least this long

    return {
        "events": float(len(droughts)),
        "current": float(current),
        "percentile": percentile,
        "exceedance": exceedance,
        "median": float(s.median()),
        "mean": float(s.mean()),
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "max": mx,
    }

def _apply_due_to_priors(
    priors: Dict[str, float],
    double_due: Optional[Dict[str, float]],
    triple_due: Optional[Dict[str, float]],
    strength_double: float,
    strength_triple: float,
) -> Dict[str, float]:
    """Adjust structure priors using due-percentiles (optional).

    Multipliers are centered at 1.0 when percentile=0.5:
      multiplier = 1 + strength * (percentile - 0.5) * 2
    Clamped to keep things stable.
    """
    adj = dict(priors)
    def mult(p: float, strength: float) -> float:
        m = 1.0 + float(strength) * (float(p) - 0.5) * 2.0
        return max(0.10, min(3.00, m))

    if double_due and strength_double > 0:
        adj["Double"] = float(adj.get("Double", 0.0)) * mult(double_due["percentile"], strength_double)
    if triple_due and strength_triple > 0:
        adj["Triple"] = float(adj.get("Triple", 0.0)) * mult(triple_due["percentile"], strength_triple)

    # Renormalize (avoid division by 0)
    total = sum(max(v, 0.0) for v in adj.values())
    if total <= 0:
        return priors
    return {k: float(max(v, 0.0) / total) for k, v in adj.items()}

def _compute_overdue_distances(history: List[str]) -> List[Dict[int, int]]:
    n = len(history)
    last_seen = [{d: None for d in DIGITS} for _ in range(3)]
    for back, idx in enumerate(range(n - 1, -1, -1)):
        a, b, c = _digits_of(history[idx])
        for pos, dig in enumerate((a, b, c)):
            if last_seen[pos][dig] is None:
                last_seen[pos][dig] = back

    overdue = []
    for pos in range(3):
        dist = {}
        for d in DIGITS:
            dist[d] = last_seen[pos][d] if last_seen[pos][d] is not None else (n + 1)
        overdue.append(dist)
    return overdue

def _build_due_grid(history: List[str], lookback: int, rows: int) -> Tuple[Grid, List[Dict[int, int]]]:
    if len(history) < lookback:
        raise ValueError(f"Need at least {lookback} draws to build the grid (you provided {len(history)}).")

    overdue = _compute_overdue_distances(history)
    recent = history[-lookback:]
    seen_by_pos = [set() for _ in range(3)]
    for s in recent:
        a, b, c = _digits_of(s)
        seen_by_pos[0].add(a)
        seen_by_pos[1].add(b)
        seen_by_pos[2].add(c)

    cols: List[List[int]] = []
    for pos in range(3):
        due = [d for d in DIGITS if d not in seen_by_pos[pos]]
        due_sorted = sorted(due, key=lambda d: (-overdue[pos][d], d))
        picked = due_sorted[:rows]
        if len(picked) < rows:
            remaining = [d for d in DIGITS if d not in picked]
            remaining_sorted = sorted(remaining, key=lambda d: (-overdue[pos][d], d))
            picked += remaining_sorted[: (rows - len(picked))]
        cols.append(picked[:rows])

    cells = [[cols[c][r] for c in range(3)] for r in range(rows)]
    return Grid(rows=rows, cols=3, cells=cells), overdue

def _adjacency_strength(grid: Grid, include_diagonal: bool) -> Dict[Tuple[int, int], int]:
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if include_diagonal:
        dirs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    edges: Dict[Tuple[int, int], int] = {}
    for r in range(grid.rows):
        for c in range(grid.cols):
            d1 = grid.cells[r][c]
            for dr, dc in dirs:
                rr, cc = r + dr, c + dc
                if 0 <= rr < grid.rows and 0 <= cc < grid.cols:
                    d2 = grid.cells[rr][cc]
                    key = (d1, d2) if d1 == d2 else (min(d1, d2), max(d1, d2))
                    edges[key] = edges.get(key, 0) + 1

    norm: Dict[Tuple[int, int], int] = {}
    for (a, b), cnt in edges.items():
        norm[(a, b)] = cnt // 2
    return norm

def _neighbors_mod10(d: int) -> Set[int]:
    return {(d - 1) % 10, (d + 1) % 10}

def _doubles_support_sets(prev: str, lookback_hist: List[str]) -> Tuple[Set[int], Set[int], Set[int]]:
    pa, pb, pc = _digits_of(prev)
    carry = {pa, pb, pc}
    neighbors = set()
    for d in carry:
        neighbors |= _neighbors_mod10(d)

    cnt = {d: 0 for d in DIGITS}
    for s in lookback_hist:
        a, b, c = _digits_of(s)
        cnt[a] += 1
        cnt[b] += 1
        cnt[c] += 1
    trending = {d for d, k in cnt.items() if k >= 2}
    return carry, neighbors, trending

def _score_candidate(
    candidate: str,
    grid: Grid,
    overdue: List[Dict[int, int]],
    adj: Dict[Tuple[int, int], int],
    prev: Optional[str],
    recent_hist_for_trend: List[str],
    weights: Dict[str, float],
    structure_priors: Optional[Dict[str, float]] = None,
    structure_prior_weight: float = 0.0,
) -> Tuple[float, Dict[str, object]]:
    a, b, c = _digits_of(candidate)

    pos_score = overdue[0][a] + overdue[1][b] + overdue[2][c]

    pairs = [(a, b), (b, c), (a, c)]
    touch_sum = 0
    touch_details = []
    for x, y in pairs:
        key = (x, y) if x == y else (min(x, y), max(x, y))
        s = adj.get(key, 0)
        touch_sum += s
        touch_details.append(((x, y), s))

    row_bonus = 0.0
    row_match = None
    for r in range(grid.rows):
        row_num = f"{grid.cells[r][0]}{grid.cells[r][1]}{grid.cells[r][2]}"
        if candidate == row_num:
            row_match = r
            row_bonus = weights["row_bonus"] / (weights["row_rank_decay"] ** r)
            break

    doubles_bonus = 0.0
    doubles_info = {"is_double": False, "double_digit": None, "carry": False, "neighbor": False, "trending": False}
    if prev is not None:
        carry, neighbors, trending = _doubles_support_sets(prev, recent_hist_for_trend)
        double_digit = None
        if a == b:
            double_digit = a
        elif b == c:
            double_digit = b
        elif a == c:
            double_digit = a

        if double_digit is not None:
            doubles_info["is_double"] = True
            doubles_info["double_digit"] = double_digit
            supported = 0.0
            if double_digit in carry:
                supported += weights["carryover"]
                doubles_info["carry"] = True
            if double_digit in neighbors:
                supported += weights["neighbor"]
                doubles_info["neighbor"] = True
            if double_digit in trending:
                supported += weights["neighbor"]
                doubles_info["trending"] = True

            if supported > 0:
                doubles_bonus = weights["doubles"] + supported
            else:
                doubles_bonus = 0.0

    total = (
        weights["pos_overdue"] * pos_score
        + weights["touch"] * touch_sum
        + row_bonus
        + doubles_bonus
    )

    # Optional structure prior: shift ranking toward the most common structure
    # in the user's selected history slice (Single / Double / Triple).
    stype = _structure_type(candidate)
    prior_bonus = 0.0
    prior_p = None
    if structure_priors and structure_prior_weight > 0:
        prior_p = float(structure_priors.get(stype, 0.0))
        nonzero = [v for v in structure_priors.values() if v and v > 0]
        min_p = min(nonzero) if nonzero else 1e-9
        prior_bonus = float(structure_prior_weight) * (math.log(prior_p + 1e-9) - math.log(min_p + 1e-9))
        total += prior_bonus


    explain = {
        "pos_overdue_sum": pos_score,
        "touch_sum": touch_sum,
        "touch_details": touch_details,
        "row_match": row_match,
        "row_bonus": row_bonus,
        "doubles": doubles_info,
        "total_score": total,
        "structure_type": stype,
        "structure_prior_p": prior_p,
        "structure_prior_bonus": prior_bonus,
    }
    return total, explain

def _generate_candidates_from_grid(grid: Grid, include_cross_product: bool, include_doubles_expansion: bool, include_triples_expansion: bool) -> Set[str]:
    col0 = [grid.cells[r][0] for r in range(grid.rows)]
    col1 = [grid.cells[r][1] for r in range(grid.rows)]
    col2 = [grid.cells[r][2] for r in range(grid.rows)]
    digits_any = sorted(set(col0 + col1 + col2))

    cands: Set[str] = set()

    for r in range(grid.rows):
        cands.add(f"{grid.cells[r][0]}{grid.cells[r][1]}{grid.cells[r][2]}")

    if include_cross_product:
        for a, b, c in itertools.product(col0, col1, col2):
            cands.add(f"{a}{b}{c}")

    if include_doubles_expansion:
        for d in digits_any:
            for x in digits_any:
                cands.add(f"{d}{d}{x}")
                cands.add(f"{d}{x}{d}")
                cands.add(f"{x}{d}{d}")


    if include_triples_expansion:
        for d in digits_any:
            cands.add(f"{d}{d}{d}")

    return {s.zfill(3) for s in cands}

def _render_grid(grid: Grid) -> pd.DataFrame:
    df = pd.DataFrame(grid.cells, columns=["Pos1 (Hundreds)", "Pos2 (Tens)", "Pos3 (Ones)"])
    df.index = [f"Row {i+1}" for i in range(grid.rows)]
    return df

def _hit_touching_pair_in_winner(grid: Grid, winner: str, include_diagonal: bool) -> Dict[str, object]:
    adj = _adjacency_strength(grid, include_diagonal)
    a, b, c = _digits_of(winner)
    pairs = [(a, b), (b, c), (a, c)]
    touches = []
    for x, y in pairs:
        key = (x, y) if x == y else (min(x, y), max(x, y))
        s = adj.get(key, 0)
        touches.append((x, y, s))
    any_touch = any(s > 0 for _, _, s in touches)
    all_digits_in_grid = set([a, b, c]).issubset(grid.digits_set())
    return {"any_touch": any_touch, "touch_details": touches, "all_digits_in_grid": all_digits_in_grid}

def main():
    st.set_page_config(page_title="Pick-3 Touch Grid Predictor", layout="wide")
    st.title("Pick-3 Touch Grid Predictor (Due-Grid + Touching-Pairs + Doubles Overlay)")

    st.markdown(
        """
This app recreates the **positional due-digit grid** approach (like the chart you shared), then ranks Pick-3 triplets.

**Core idea**
- Build a grid from the last *N* draws: digits **missing** in each position (hundreds/tens/ones), ranked by **overdue distance**.
- Rank candidates higher when they:
  - Have **touching pairs** (horizontal/vertical/diagonal) inside the grid
  - Match a **direct grid row**
  - (Optional) Fit a **doubles overlay** tied to the previous winner (carryover / +/- 1 / recent-trending)

No method is guaranteed. This is a scoring/ranking tool.
        """
    )

    with st.sidebar:
        st.header("Inputs")

        data_mode = st.radio("Data input type", ["Paste results (simple)", "Upload file (LotteryPost TXT/CSV)"], index=0)
        reverse_order = st.checkbox("My pasted results are newest -> oldest (reverse them)", value=False)
        draw_time = st.selectbox("Draw time (optional)", ["(ignore)", "Morning", "Day", "Evening", "Night"], index=0)

        lookback = st.number_input("Lookback draws for grid (N)", min_value=3, max_value=30, value=4, step=1)
        grid_rows = st.number_input("Grid rows", min_value=3, max_value=7, value=4, step=1)

        include_diagonal = st.checkbox("Count diagonal (catacorner) touching", value=True)

        include_cross_product = st.checkbox("Generate full column cross-product candidates (rows^3)", value=True)
        include_doubles_expansion = st.checkbox("Add doubles expansions from grid digits (AAB/ABA/BAA)", value=True)
        include_triples_expansion = st.checkbox("Include triples (AAA) from grid digits", value=False)


        st.subheader("Scoring weights")
        weights = DEFAULT_WEIGHTS.copy()
        weights["pos_overdue"] = st.slider("Positional overdue weight", 0.0, 5.0, float(weights["pos_overdue"]), 0.1)
        weights["touch"] = st.slider("Touching-pair weight", 0.0, 10.0, float(weights["touch"]), 0.1)
        weights["row_bonus"] = st.slider("Direct-row bonus", 0.0, 20.0, float(weights["row_bonus"]), 0.5)
        weights["row_rank_decay"] = st.slider("Row bonus decay (higher = faster drop)", 1.0, 3.0, float(weights["row_rank_decay"]), 0.05)
        weights["doubles"] = st.slider("Doubles support bonus", 0.0, 15.0, float(weights["doubles"]), 0.5)
        weights["carryover"] = st.slider("Carryover digit add-on", 0.0, 10.0, float(weights["carryover"]), 0.5)
        weights["neighbor"] = st.slider("+/- 1 neighbor add-on", 0.0, 10.0, float(weights["neighbor"]), 0.5)

        st.subheader("Structure likelihood (optional)")
        prior_window = st.selectbox("Structure prior window", ["All history", "Last 50", "Last 100", "Last 365"], index=0)
        structure_prior_weight = st.slider("Structure prior strength (0 = off)", 0.0, 20.0, 0.0, 0.5)


        st.subheader("Structure due meter")
        due_window = st.selectbox("Due meter history window", ["All history", "Last 100", "Last 365", "Last 730"], index=0)
        apply_due_to_ranking = st.checkbox("Use due meter to adjust structure priors", value=False)
        col_due1, col_due2 = st.columns(2)
        with col_due1:
            double_due_strength = st.slider("Double due influence", 0.0, 2.0, 0.0, 0.1)
        with col_due2:
            triple_due_strength = st.slider("Triple due influence", 0.0, 2.0, 0.0, 0.1)
        due_meter_box = st.empty()

        st.subheader("Backtest")
        do_backtest = st.checkbox("Run backtest (build grid -> test next draw) on this dataset", value=False)
        backtest_max = st.number_input("Max backtest steps (0 = all)", min_value=0, max_value=5000, value=0, step=50)

    history: List[str] = []
    df_full = None

    if data_mode == "Paste results (simple)":
        txt = st.text_area(
            "Paste Pick-3 results (any text; app extracts 123 or 1-2-3 formats)",
            height=200,
            placeholder="Example:\n746\n746\n676\n084\n043\n278\n(or paste a block copied from a site)",
        )
        if txt.strip():
            history = _parse_simple_numbers(txt, reverse_order)
    else:
        up = st.file_uploader("Upload file (CSV / TXT / TSV)", type=["csv", "txt", "tsv"])
        if up is not None:
            name = (up.name or "").lower()
            if name.endswith(".csv"):
                df = pd.read_csv(up)
                df_full = _parse_csv(df)
            else:
                raw = up.getvalue().decode("utf-8", errors="ignore")
                df_full = _parse_lotterypost_export_text(raw)

    if data_mode == "Upload file (LotteryPost TXT/CSV)" and df_full is not None:
        df_use = df_full.copy()
        if draw_time != "(ignore)":
            df_use = df_use[df_use["draw_time"].astype(str) == draw_time].copy()

        if "date" in df_use.columns:
            df_use = df_use.sort_values("date")

        history = df_use["result"].astype(str).tolist()

        if reverse_order:
            history = list(reversed(history))

    if not history:
        st.info("Add results to begin. You can paste results or upload a file.")
        st.stop()

    history = [s for s in history if _ensure_3digits(s) is not None]
    if len(history) < int(lookback) + 1:
        st.warning(f"Need at least {int(lookback)+1} results to build the grid and score doubles. You have {len(history)}.")
        st.stop()

    prev_winner = history[-1]
    exclude_most_recent = st.checkbox("Exclude most recent result when building the grid (use previous N draws)", value=False)
    grid_history = history[:-1] if exclude_most_recent else history

    grid, overdue = _build_due_grid(grid_history, lookback=int(lookback), rows=int(grid_rows))
    adj = _adjacency_strength(grid, include_diagonal=include_diagonal)
    recent_for_trend = history[-int(lookback):]

    # Compute empirical structure probabilities for optional structure-prior scoring
    prior_hist = history[:]
    if prior_window == "Last 50":
        prior_hist = prior_hist[-50:]
    elif prior_window == "Last 100":
        prior_hist = prior_hist[-100:]
    elif prior_window == "Last 365":
        prior_hist = prior_hist[-365:]
    structure_priors = _compute_structure_priors(prior_hist)

    # --- Structure Due Meter (optional) ---
    due_hist = history[:]
    if due_window == "Last 100":
        due_hist = due_hist[-100:]
    elif due_window == "Last 365":
        due_hist = due_hist[-365:]
    elif due_window == "Last 730":
        due_hist = due_hist[-730:]

    double_due = _due_metrics_from_history(due_hist, target="Double")
    triple_due = _due_metrics_from_history(due_hist, target="Triple")

    # Show due meter in the sidebar
    with due_meter_box.container():
        st.caption("Due meter is based on how long it's been since the last Double/Triple (within the selected draw-time stream).")
        if double_due:
            st.write(
                f"**Double drought:** {int(double_due['current'])} draws • "
                f"~{double_due['percentile']*100:.0f}th percentile • "
                f"p90≈{double_due['p90']:.0f}, max={double_due['max']:.0f} (events={int(double_due['events'])})"
            )
        else:
            st.write("**Double drought:** not enough Double events in the selected due window (need ≥5).")
        if triple_due:
            st.write(
                f"**Triple drought:** {int(triple_due['current'])} draws • "
                f"~{triple_due['percentile']*100:.0f}th percentile • "
                f"p90≈{triple_due['p90']:.0f}, max={triple_due['max']:.0f} (events={int(triple_due['events'])})"
            )
        else:
            st.write("**Triple drought:** not enough Triple events in the selected due window (need ≥5).")

    structure_priors_base = dict(structure_priors)
    if apply_due_to_ranking and (double_due_strength > 0 or triple_due_strength > 0):
        structure_priors = _apply_due_to_priors(
            structure_priors,
            double_due=double_due,
            triple_due=triple_due,
            strength_double=double_due_strength,
            strength_triple=triple_due_strength,
        )


    left, right = st.columns([1, 1.4])

    with left:
        st.subheader("Prediction Grid (positional due digits)")
        st.dataframe(_render_grid(grid), use_container_width=True)
        st.caption("Columns are position-specific. Rows are the most-overdue digits per position.")

        st.subheader("Grid digits & touching pairs (summary)")
        st.write(f"Unique digits in grid: **{sorted(grid.digits_set())}**")
        st.write(f"Unique touching pairs (incl. diagonal={include_diagonal}): **{len([k for k,v in adj.items() if v>0])}**")
        if apply_due_to_ranking and (double_due_strength > 0 or triple_due_strength > 0):
            st.write(
                f"Structure prior (base {prior_window}): **Single {structure_priors_base['Single']:.1%} / "
                f"Double {structure_priors_base['Double']:.1%} / Triple {structure_priors_base['Triple']:.1%}**"
            )
            st.write(
                f"Structure prior (after due adjust): **Single {structure_priors['Single']:.1%} / "
                f"Double {structure_priors['Double']:.1%} / Triple {structure_priors['Triple']:.1%}**"
            )
        else:
            st.write(
                f"Structure prior ({prior_window}): **Single {structure_priors['Single']:.1%} / "
                f"Double {structure_priors['Double']:.1%} / Triple {structure_priors['Triple']:.1%}**"
            )

    candidates = sorted(_generate_candidates_from_grid(grid, include_cross_product, include_doubles_expansion, include_triples_expansion))
    scored = []
    for cand in candidates:
        score, explain = _score_candidate(
            cand,
            grid=grid,
            overdue=overdue,
            adj=adj,
            prev=prev_winner,
            recent_hist_for_trend=recent_for_trend,
            weights=weights,
            structure_priors=structure_priors,
            structure_prior_weight=structure_prior_weight,
        )
        scored.append((score, cand, explain))
    scored.sort(key=lambda x: (-x[0], x[1]))

    with right:
        st.subheader("Ranked triplets (most likely -> least likely)")
        top_n = st.number_input("Show top N", min_value=10, max_value=min(500, len(scored)), value=min(100, len(scored)), step=10)
        rows_out = []
        for rank, (score, cand, explain) in enumerate(scored[: int(top_n)], start=1):
            tags = []
            if explain["row_match"] is not None:
                tags.append(f"ROW{explain['row_match']+1}")
            if explain["touch_sum"] > 0:
                tags.append(f"TOUCH({explain['touch_sum']})")
            if explain["doubles"]["is_double"]:
                dd = explain["doubles"]["double_digit"]
                sub = []
                if explain["doubles"]["carry"]:
                    sub.append("carry")
                if explain["doubles"]["neighbor"]:
                    sub.append("+/-1")
                if explain["doubles"]["trending"]:
                    sub.append("trend")
                tags.append(f"DOUBLE({dd}:{'/'.join(sub) if sub else 'none'})")

            rows_out.append({
                "Rank": rank,
                "Triplet": cand,
                "Structure": explain.get("structure_type",""),
                "Score": round(float(score), 2),
                "Tags": ", ".join(tags),
                "PosOverdue": int(explain["pos_overdue_sum"]),
                "Touch": int(explain["touch_sum"]),
                "RowMatch": "" if explain["row_match"] is None else (int(explain["row_match"]) + 1),
            })

        out_df = pd.DataFrame(rows_out)
        st.dataframe(out_df, use_container_width=True)

        st.download_button(
            "Download ranked list (CSV)",
            data=out_df.to_csv(index=False).encode("utf-8"),
            file_name="pick3_ranked_triplets.csv",
            mime="text/csv",
        )

        st.subheader("Explain a specific triplet")
        pick = st.text_input("Enter a 3-digit triplet to explain (e.g., 168)", value="")
        pick = _ensure_3digits(pick) if pick else None
        if pick:
            score, explain = _score_candidate(
                pick,
                grid=grid,
                overdue=overdue,
                adj=adj,
                prev=prev_winner,
                recent_hist_for_trend=recent_for_trend,
                weights=weights,
            structure_priors=structure_priors,
            structure_prior_weight=structure_prior_weight,
            )
            st.write(f"**{pick}** score = **{round(float(score),2)}**")
            st.json(explain)

    if do_backtest:
        st.header("Backtest results")
        hist = history[:]
        start = int(lookback)
        steps = list(range(start, len(hist)))
        if int(backtest_max) and int(backtest_max) > 0:
            steps = steps[-int(backtest_max):]

        results = []
        for i in steps:
            prior = hist[:i]
            winner = hist[i]
            try:
                g, _od = _build_due_grid(prior, lookback=int(lookback), rows=int(grid_rows))
            except Exception:
                continue
            hit = _hit_touching_pair_in_winner(g, winner, include_diagonal=include_diagonal)
            wa, wb, wc = _digits_of(winner)
            is_double = (wa == wb) or (wb == wc) or (wa == wc)
            prev = hist[i - 1]
            carry, neigh, trend = _doubles_support_sets(prev, prior[-int(lookback):])

            double_digit = None
            if is_double:
                if wa == wb:
                    double_digit = wa
                elif wb == wc:
                    double_digit = wb
                elif wa == wc:
                    double_digit = wa

            results.append({
                "index": i,
                "prev": prev,
                "winner": winner,
                "is_double": is_double,
                "double_digit": "" if double_digit is None else double_digit,
                "any_touch_pair": hit["any_touch"],
                "all_digits_in_grid": hit["all_digits_in_grid"],
                "double_digit_in_grid": (double_digit in g.digits_set()) if double_digit is not None else False,
                "double_digit_in_prev": (double_digit in set(_digits_of(prev))) if double_digit is not None else False,
                "double_digit_is_neighbor": (double_digit in neigh) if double_digit is not None else False,
            })

        if not results:
            st.warning("Backtest produced no rows (check inputs).")
            st.stop()

        bt = pd.DataFrame(results)
        total = len(bt)
        st.write(f"Backtest rows: **{total}**")
        st.write(f"Winner had >= 1 touching pair in grid: **{bt['any_touch_pair'].mean():.1%}**")
        st.write(f"All winner digits appeared somewhere in grid: **{bt['all_digits_in_grid'].mean():.1%}**")

        doubles = bt[bt["is_double"] == True].copy()
        if len(doubles) > 0:
            st.subheader("Doubles diagnostics")
            st.write(f"Doubles count: **{len(doubles)}** ({len(doubles)/total:.1%} of tested draws)")
            st.write(f"Doubled digit appears in grid: **{doubles['double_digit_in_grid'].mean():.1%}**")
            st.write(f"Doubled digit appears in previous winner: **{doubles['double_digit_in_prev'].mean():.1%}**")
            st.write(f"Doubled digit is +/- 1 neighbor of a previous digit: **{doubles['double_digit_is_neighbor'].mean():.1%}**")
            st.dataframe(doubles.tail(50), use_container_width=True)
        else:
            st.info("No doubles found in the backtested slice.")

        st.subheader("Backtest table (last 100 rows)")
        st.dataframe(bt.tail(100), use_container_width=True)

if __name__ == "__main__":
    main()
