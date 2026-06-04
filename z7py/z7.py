"""
Z7 (IGEO7/GBT) discrete global grid system — Python/Numba implementation.

The core of this implementation is a bit-packed 64-bit representation of Z7 indices,
optimized for fast neighbor traversal and range arithmetic using Numba.

Bit layout of a packed UInt64 Z7 index:
  bits 63-60  : base cell ID (0-11)
  bits 59-57  : resolution digit 1 (3 bits, values 0-6)
  bits 56-54  : resolution digit 2
  ...
  bits  2-0   : resolution digit 20
  
  Note: A digit value of 7 indicates 'beyond resolution' (padding).
  This layout allows for efficient bit-shifting to extract digits and 
  maintains lexicographical sorting consistency.
"""

from __future__ import annotations

import numpy as np
import numba as nb
from numba import uint8, uint64, int64, boolean

# ---------------------------------------------------------------------------
# Resolution statistics (pure-Python dict — not called in hot paths)
# ---------------------------------------------------------------------------

RESOLUTION_STATS: dict[int, dict] = {
    0:  dict(num_cells=12,                    area_km2=51006562.1724089,  area_m2=51006562172408.9,  cls_km=8199.5003701,  cls_m=8199500.3701),
    1:  dict(num_cells=72,                    area_km2=7286651.7389156,   area_m2=7286651738915.6,   cls_km=3053.2232428,  cls_m=3053223.2428),
    2:  dict(num_cells=492,                   area_km2=1040950.2484165,   area_m2=1040950248416.5,   cls_km=1151.6430095,  cls_m=1151643.0095),
    3:  dict(num_cells=3432,                  area_km2=148707.1783452,    area_m2=148707178345.2,    cls_km=435.1531492,   cls_m=435153.1492),
    4:  dict(num_cells=24012,                 area_km2=21243.8826207,     area_m2=21243882620.7,     cls_km=164.4655799,   cls_m=164465.5799),
    5:  dict(num_cells=168072,                area_km2=3034.8403744,      area_m2=3034840374.4,      cls_km=62.1617764,    cls_m=62161.7764),
    6:  dict(num_cells=1176492,               area_km2=433.5486249,       area_m2=433548624.9,       cls_km=23.4949231,    cls_m=23494.9231),
    7:  dict(num_cells=8235432,               area_km2=61.9355178,        area_m2=61935517.8,        cls_km=8.8802451,     cls_m=8880.2451),
    8:  dict(num_cells=57648012,              area_km2=8.8479311,         area_m2=8847931.1,         cls_km=3.3564171,     cls_m=3356.4171),
    9:  dict(num_cells=403536072,             area_km2=1.2639902,         area_m2=1263990.2,         cls_km=1.2686064,     cls_m=1268.6064),
    10: dict(num_cells=2824752492,            area_km2=0.1805700,         area_m2=180570.0,          cls_km=0.4794882,     cls_m=479.4882),
    11: dict(num_cells=19773267432,           area_km2=0.0257957,         area_m2=25795.7,           cls_km=0.1812295,     cls_m=181.2295),
    12: dict(num_cells=138412872012,          area_km2=0.0036851,         area_m2=3685.1,            cls_km=0.0684983,     cls_m=68.4983),
    13: dict(num_cells=968890104072,          area_km2=0.0005264,         area_m2=526.4,             cls_km=0.0258899,     cls_m=25.8899),
    14: dict(num_cells=6782230728492,         area_km2=0.0000752,         area_m2=75.2,              cls_km=0.0097855,     cls_m=9.7855),
    15: dict(num_cells=47475615099432,        area_km2=0.0000107,         area_m2=10.7,              cls_km=0.0036986,     cls_m=3.6986),
    16: dict(num_cells=332329305696012,       area_km2=0.0000015,         area_m2=1.5,               cls_km=0.0013979,     cls_m=1.3979),
    17: dict(num_cells=2326305139872072,      area_km2=0.0000002,         area_m2=0.2,               cls_km=0.0005284,     cls_m=0.5284),
    18: dict(num_cells=16284135979104492,     area_km2=0.0,               area_m2=0.0,               cls_km=0.0001997,     cls_m=0.1997),
    19: dict(num_cells=113988951853731432,    area_km2=0.0,               area_m2=0.0,               cls_km=0.0000755,     cls_m=0.0755),
    20: dict(num_cells=797922662976120012,    area_km2=0.0,               area_m2=0.0,               cls_km=0.0000285,     cls_m=0.0285),
}

# Flat numpy arrays for numba-accessible resolution lookups (index = resolution)
_NUM_CELLS   = np.array([RESOLUTION_STATS[r]["num_cells"]   for r in range(21)], dtype=np.int64)
_AREA_KM2    = np.array([RESOLUTION_STATS[r]["area_km2"]    for r in range(21)], dtype=np.float64)
_AREA_M2     = np.array([RESOLUTION_STATS[r]["area_m2"]     for r in range(21)], dtype=np.float64)
_CLS_KM      = np.array([RESOLUTION_STATS[r]["cls_km"]      for r in range(21)], dtype=np.float64)
_CLS_M       = np.array([RESOLUTION_STATS[r]["cls_m"]       for r in range(21)], dtype=np.float64)


def get_resolution_stats(resolution: int) -> dict:
    if not (0 <= resolution <= 20):
        raise ValueError(f"Resolution must be 0-20, got {resolution}")
    return RESOLUTION_STATS[resolution]

def get_num_cells(resolution: int) -> int:
    return get_resolution_stats(resolution)["num_cells"]

def get_cell_area_m2(resolution: int) -> float:
    return get_resolution_stats(resolution)["area_m2"]

def get_cell_area_km2(resolution: int) -> float:
    return get_resolution_stats(resolution)["area_km2"]

def get_cls_m(resolution: int) -> float:
    return get_resolution_stats(resolution)["cls_m"]

def get_cls_km(resolution: int) -> float:
    return get_resolution_stats(resolution)["cls_km"]


def find_resolution_by_value(target: float, metric: str, prefer: str = "closest") -> int:
    valid_metrics = ("num_cells", "area_km2", "area_m2", "cls_km", "cls_m")
    if metric not in valid_metrics:
        raise ValueError(f"Metric must be one of {valid_metrics}")
    if prefer not in ("closest", "larger", "smaller"):
        raise ValueError("prefer must be 'closest', 'larger', or 'smaller'")

    best_res = None
    best_diff = float("inf")

    for res in range(21):
        value = RESOLUTION_STATS[res][metric]
        if prefer == "closest":
            diff = abs(value - target)
            if diff < best_diff:
                best_diff = diff
                best_res = res
        elif prefer == "larger":
            if value >= target:
                diff = value - target
                if diff < best_diff:
                    best_diff = diff
                    best_res = res
        else:  # smaller
            if value <= target:
                diff = target - value
                if diff < best_diff:
                    best_diff = diff
                    best_res = res

    return best_res


def find_resolution_by_cls_m(target_m: float, prefer: str = "closest") -> int:
    return find_resolution_by_value(target_m, "cls_m", prefer=prefer)

def find_resolution_by_area_m2(target_m2: float, prefer: str = "closest") -> int:
    return find_resolution_by_value(target_m2, "area_m2", prefer=prefer)

def find_resolution_by_num_cells(target_cells: int, prefer: str = "closest") -> int:
    return find_resolution_by_value(target_cells, "num_cells", prefer=prefer)


# ---------------------------------------------------------------------------
# Lookup tables as contiguous numpy arrays (accessible inside @njit functions)
# ---------------------------------------------------------------------------

# BASE_CELL_NEIGHBOURS[base_cell, direction_idx (0-5)] — 6 slots per cell
# (Julia 1-indexed, Python 0-indexed)
_BASE_CELL_NEIGHBOURS_RAW = np.array([
    [5, 4, 4, 2, 1, 3],   # base cell 0
    [5, 0, 0, 6, 10, 2],  # base cell 1
    [1, 0, 0, 7, 6, 3],   # base cell 2
    [2, 0, 0, 8, 7, 4],   # base cell 3
    [3, 0, 0, 9, 8, 5],   # base cell 4
    [4, 0, 0, 10, 9, 1],  # base cell 5
    [10, 2, 1, 11, 11, 7],# base cell 6
    [6, 3, 2, 11, 11, 8], # base cell 7
    [7, 4, 3, 11, 11, 9], # base cell 8
    [8, 5, 4, 11, 11, 10],# base cell 9
    [9, 1, 5, 11, 11, 6], # base cell 10
    [9, 6, 10, 8, 8, 7],  # base cell 11
], dtype=np.uint8)

# Exclusion index (1-based in Julia → 0-based here)
_EXCLUDE_NEIGHBOURS = np.array([2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5], dtype=np.uint8)

_ROTATIONS = np.array([0, 5, 0, 1, 3, 4, 5, 4, 3, 1, 0, 0], dtype=np.uint8)

_POLE_0_ROTATIONS = np.array([
    [0, 1, 0, 1, 0, 2],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 2, 2],
    [5, 5, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [5, 0, 4, 0, 4, 0],
], dtype=np.uint8)

# GBT addition tables (7×7)
_GBT_CW_0 = np.array([
    [0, 1, 2, 3, 4, 5, 6],
    [1, 4, 3, 6, 5, 2, 0],
    [2, 3, 1, 4, 6, 0, 5],
    [3, 6, 4, 5, 0, 1, 2],
    [4, 5, 6, 0, 2, 3, 1],
    [5, 2, 0, 1, 3, 6, 4],
    [6, 0, 5, 2, 1, 4, 3],
], dtype=np.uint8)

_GBT_CW_1 = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 5, 0],
    [0, 0, 2, 3, 0, 0, 2],
    [0, 1, 3, 3, 0, 0, 0],
    [0, 0, 0, 0, 4, 4, 6],
    [0, 5, 0, 0, 4, 5, 0],
    [0, 0, 2, 0, 6, 0, 6],
], dtype=np.uint8)

_GBT_CCW_0 = np.array([
    [0, 1, 2, 3, 4, 5, 6],
    [1, 2, 3, 4, 5, 6, 0],
    [2, 3, 4, 5, 6, 0, 1],
    [3, 4, 5, 6, 0, 1, 2],
    [4, 5, 6, 0, 1, 2, 3],
    [5, 6, 0, 1, 2, 3, 4],
    [6, 0, 1, 2, 3, 4, 5],
], dtype=np.uint8)

_GBT_CCW_1 = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 3, 0, 1, 0],
    [0, 0, 2, 2, 0, 0, 6],
    [0, 3, 2, 3, 0, 0, 0],
    [0, 0, 0, 0, 4, 5, 4],
    [0, 1, 0, 0, 5, 5, 0],
    [0, 0, 6, 0, 4, 0, 6],
], dtype=np.uint8)

_MOD_7_TABLE = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6], dtype=np.uint8)

# ---------------------------------------------------------------------------
# Core bit-manipulation helpers — compiled with @njit for speed
# ---------------------------------------------------------------------------

@nb.njit(cache=True)
def get_base_cell(raw: np.uint64) -> np.uint8:
    """Extract the base cell ID (0-11) from the top 4 bits."""
    return np.uint8((raw >> np.uint64(60)) & np.uint64(0x0F))


def _ensure_u64(raw) -> np.uint64:
    """Safety wrapper to ensure values are treated as unsigned 64-bit integers."""
    return np.uint64(raw & 0xFFFFFFFFFFFFFFFF) if isinstance(raw, int) else np.uint64(raw)


@nb.njit(cache=True)
def get_digit(raw: np.uint64, i: int) -> np.uint8:
    """
    Extract a single resolution digit (0-6).
    i is the 1-based position (1 to 20).
    Each digit occupies 3 bits, starting after the 4-bit base cell.
    """
    shift = np.uint64(57 - 3 * (i - 1))
    return np.uint8((raw >> shift) & np.uint64(0x07))


@nb.njit(cache=True)
def get_digits(raw: np.uint64) -> np.ndarray:
    """Return all 20 digits as a uint8 array."""
    out = np.empty(20, dtype=np.uint8)
    for i in range(1, 21):
        out[i - 1] = get_digit(raw, i)
    return out


@nb.njit(cache=True)
def get_resolution(raw: np.uint64) -> int:
    for i in range(1, 21):
        if get_digit(raw, i) == np.uint8(7):
            return i - 1
    return 20


@nb.njit(cache=True)
def z7_to_monotonic_int(raw: np.uint64, resolution: int) -> np.uint64:
    """
    Map a hierarchical Z7 index onto a perfectly sequential 'number line'.
    
    This is the core of the RangeIndex: it ensures that all children of a parent
    are numerically contiguous (neighbors in space become neighbors in the index).
    Value = (BaseCell * 7^L) + sum(Digit_i * 7^(L-i)).
    """
    bc = get_base_cell(raw)
    val = np.uint64(bc)
    for i in range(1, resolution + 1):
        # We multiply by 7 because each cell has exactly 7 children.
        val = val * np.uint64(7) + np.uint64(get_digit(raw, i))
    return val


@nb.njit(cache=True)
def monotonic_int_to_z7(val: np.uint64, resolution: int) -> np.uint64:
    """
    Inverse of z7_to_monotonic_int: reconstruct the bit-packed Z7 ID from a number line position.
    Useful for 'forward' mapping in an Xarray index.
    """
    v = val
    digits = np.empty(resolution, dtype=np.uint8)
    for i in range(resolution - 1, -1, -1):
        digits[i] = np.uint8(v % np.uint64(7))
        v //= np.uint64(7)
    
    base_cell = np.uint8(v)
    return _encode_z7int_jit(base_cell, digits)


@nb.njit(cache=True)
def _encode_z7int_jit(base_cell: np.uint8, digits: np.ndarray) -> np.uint64:
    """Encode base cell + digits array (up to 20 uint8) into a packed UInt64."""
    result = np.uint64(base_cell) << np.uint64(60)
    n = min(len(digits), 20)
    for i in range(n):
        result |= np.uint64(digits[i]) << np.uint64(57 - 3 * i)
    for i in range(n, 20):
        result |= np.uint64(7) << np.uint64(57 - 3 * i)
    return result


def encode_z7int(base_cell, digits) -> np.uint64:
    """Python wrapper: accepts any integer types, always returns np.uint64."""
    bc = np.uint8(int(base_cell))
    dg = np.asarray(digits, dtype=np.uint8)
    return np.uint64(_encode_z7int_jit(bc, dg))


def decode_z7int(raw) -> tuple:
    """Return (base_cell: uint8, digits: uint8[20])."""
    raw = np.uint64(raw)
    base_cell = np.uint8((raw >> np.uint64(60)) & np.uint64(0x0F))
    digits = np.array(
        [int((raw >> np.uint64(57 - 3 * i)) & np.uint64(0x07)) for i in range(20)],
        dtype=np.uint8,
    )
    return base_cell, digits


@nb.njit(cache=True)
def get_parent_at(raw: np.uint64, resolution: int) -> np.uint64:
    """Return the parent index at the given resolution."""
    if resolution < 0:
        resolution = 0
    if resolution > 20:
        resolution = 20
    # keep base cell and first `resolution` digits, fill rest with 7
    result = raw & (np.uint64(0x0F) << np.uint64(60))
    for i in range(1, resolution + 1):
        d = get_digit(raw, i)
        shift = np.uint64(57 - 3 * (i - 1))
        result |= np.uint64(d) << shift
    for i in range(resolution + 1, 21):
        shift = np.uint64(57 - 3 * (i - 1))
        result |= np.uint64(7) << shift
    return result


def get_parent(raw, resolution: int = -1) -> np.uint64:
    """Return parent index. If resolution=-1, go one level up."""
    raw = np.uint64(raw)
    if resolution == -1:
        res = get_resolution(raw)
        resolution = max(0, res - 1)
    return np.uint64(get_parent_at(raw, resolution))


# ---------------------------------------------------------------------------
# GBT addition tables — numba-accessible
# ---------------------------------------------------------------------------

@nb.njit(cache=True)
def neighbour_addition_cw(a: np.uint8, b: np.uint8, cw0=_GBT_CW_0, cw1=_GBT_CW_1):
    return cw1[a, b], cw0[a, b]


@nb.njit(cache=True)
def neighbour_addition_ccw(a: np.uint8, b: np.uint8, ccw0=_GBT_CCW_0, ccw1=_GBT_CCW_1):
    return ccw1[a, b], ccw0[a, b]


@nb.njit(cache=True)
def neighbour_addition_ccw_mod(a: np.uint8, b: np.uint8,
                                ccw1=_GBT_CCW_1, mod7=_MOD_7_TABLE):
    return ccw1[a, b], mod7[int(a) + int(b)]


# ---------------------------------------------------------------------------
# first_non_zero — returns 1-based position of first non-7, non-0 digit
# ---------------------------------------------------------------------------

@nb.njit(cache=True)
def first_non_zero(raw: np.uint64) -> int:
    """
    Return the 1-based position of the first non-zero resolution digit
    (including 7-padding digits, which count as non-zero).
    Returns 0 if there are no resolution digits (digit 1 == 7).

    Equivalent to Julia's leading_zeros-based implementation:
    the first non-zero bit after the base-cell bits corresponds to
    either the first non-zero resolution digit or the first 7-padding digit.
    """
    if get_digit(raw, 1) == np.uint8(7):
        return 0
    for i in range(1, 21):
        if get_digit(raw, i) != np.uint8(0):
            return i
    return 0


# ---------------------------------------------------------------------------
# Single-neighbour computation (inner hot path)
# ---------------------------------------------------------------------------

@nb.njit(cache=True)
def get_neighbour(raw: np.uint64, direction: np.uint8, resolution: int,
                  cw0=_GBT_CW_0, cw1=_GBT_CW_1,
                  ccw0=_GBT_CCW_0, ccw1=_GBT_CCW_1):
    """
    Compute one neighbour in direction (1-6) using GBT (Generalized Balanced Ternary) arithmetic.
    This logic handles 'carries'—where moving in a direction requires updating parent digits.
    Returns (neighbour_raw: uint64, carry: uint8).
    """
    result = raw
    carry = np.uint8(0)

    v = get_digit(raw, resolution)
    # Z7 uses alternating CW/CCW logic per refinement level.
    is_even = (resolution % 2 == 0)

    if is_even:
        r1 = ccw1[v, direction]
        r0 = ccw0[v, direction]
    else:
        r1 = cw1[v, direction]
        r0 = cw0[v, direction]

    shift = np.uint64(57 - 3 * (resolution - 1))
    result = (result & ~(np.uint64(0x07) << shift)) | (np.uint64(r0) << shift)
    carry = r1

    # Cascade the carry up the hierarchy (like 9+1 cascading in decimal)
    if carry != np.uint8(0):
        for i in range(resolution - 1, 0, -1):
            v2 = get_digit(raw, i)
            is_even2 = (i % 2 == 0)
            if is_even2:
                r1b = ccw1[v2, carry]
                r0b = ccw0[v2, carry]
            else:
                r1b = cw1[v2, carry]
                r0b = cw0[v2, carry]
            shift2 = np.uint64(57 - 3 * (i - 1))
            result = (result & ~(np.uint64(0x07) << shift2)) | (np.uint64(r0b) << shift2)
            carry = r1b
            if carry == np.uint8(0):
                break

    return result, carry


# ---------------------------------------------------------------------------
# All-neighbours computation
# ---------------------------------------------------------------------------

_INVALID_RAW = np.uint64(np.iinfo(np.uint64).max)  # typemax(UInt64)


@nb.njit(cache=True)
def get_neighbours(raw: np.uint64,
                   bcn=_BASE_CELL_NEIGHBOURS_RAW,
                   excl=_EXCLUDE_NEIGHBOURS,
                   rotations_arr=_ROTATIONS,
                   pole0=_POLE_0_ROTATIONS,
                   cw0=_GBT_CW_0, cw1=_GBT_CW_1,
                   ccw0=_GBT_CCW_0, ccw1=_GBT_CCW_1):
    """
    Find all 6 immediate spatial neighbours of a Z7 cell.
    If a carry remains after ascending the hierarchy, it means we've crossed
    into a different base-cell (face of the ellipsoid).
    Invalid neighbours (pentagon corners) are set to UINT64_MAX.
    """
    resolution = get_resolution(raw)
    base_cell = int(get_base_cell(raw))
    exclusion = int(excl[base_cell])  # 1-based (Julia style)

    result = np.empty(6, dtype=np.uint64)
    carries = np.empty(6, dtype=np.uint8)

    # Special case: resolution 0 (base cell only)
    if resolution == 0:
        neighbour_zones = bcn[base_cell]
        for i in range(6):
            idx_1based = i + 1
            if idx_1based == exclusion:
                result[i] = _INVALID_RAW
            else:
                nb_bc = np.uint8(neighbour_zones[i])
                result[i] = _encode_z7int_jit(nb_bc, np.empty(0, dtype=np.uint8))
        return result

    # Compute 6 neighbours with carry
    neighbour_zones = bcn[base_cell]
    for d in range(1, 7):
        nb_raw, carry = get_neighbour(raw, np.uint8(d), resolution,
                                      cw0, cw1, ccw0, ccw1)
        result[d - 1] = nb_raw
        carries[d - 1] = carry

    # Handle carries (base-cell crossings)
    for i in range(6):
        carry = carries[i]
        if carry != np.uint8(0):
            new_base = np.uint64(neighbour_zones[int(carry) - 1])
            # update base cell bits
            nb_raw = (result[i] & ~(np.uint64(0x0F) << np.uint64(60))) | (new_base << np.uint64(60))

            # Rotation into polar base cells
            if new_base == np.uint64(0) or new_base == np.uint64(11):
                rots = int(rotations_arr[base_cell])
                fd = int(get_digit(raw, 1))
                if fd == 6 or fd == 1:
                    rots += 1
                if rots > 0:
                    mult = np.uint8(1)
                    for _ in range(rots):
                        mult = np.uint8((int(mult) * 5) % 7)
                    for j in range(1, resolution + 1):
                        d2 = int(get_digit(nb_raw, j))
                        rotated = np.uint64((d2 * int(mult)) % 7)
                        sh = np.uint64(57 - 3 * (j - 1))
                        nb_raw = (nb_raw & ~(np.uint64(0x07) << sh)) | (rotated << sh)

            result[i] = nb_raw

            # Rotation from polar base cells
            if base_cell == 0 or base_cell == 11:
                row = int(get_digit(raw, 1))
                col = int(get_digit(nb_raw, 1))
                if base_cell == 11:
                    row = 7 - row
                    col = 7 - col
                if 1 <= row <= 6 and 1 <= col <= 6:
                    rots2 = int(pole0[row - 1, col - 1])
                    if rots2 > 0:
                        mult2 = np.uint8(1)
                        for _ in range(rots2):
                            mult2 = np.uint8((int(mult2) * 5) % 7)
                        nb_raw2 = result[i]
                        for j in range(1, resolution + 1):
                            d2 = int(get_digit(nb_raw2, j))
                            rotated = np.uint64((d2 * int(mult2)) % 7)
                            sh = np.uint64(57 - 3 * (j - 1))
                            nb_raw2 = (nb_raw2 & ~(np.uint64(0x07) << sh)) | (rotated << sh)
                        result[i] = nb_raw2

    # Pentagon center check: all data digits 0
    data_only = (raw & ~(np.uint64(0x0F) << np.uint64(60))) >> np.uint64(3 * (20 - resolution))
    if data_only == np.uint64(0) and 1 <= exclusion <= 6:
        result[exclusion - 1] = _INVALID_RAW
        return result

    # Exclusion-zone rotation
    ref_fnz = first_non_zero(raw)
    if ref_fnz < 1:
        return result

    reference_zone = int(get_digit(raw, ref_fnz))
    multiplier = np.uint8(0)
    if (reference_zone * 5) % 7 == exclusion:
        multiplier = np.uint8(5)
    elif (reference_zone * 3) % 7 == exclusion:
        multiplier = np.uint8(3)

    if multiplier > np.uint8(0):
        for i in range(6):
            if result[i] == _INVALID_RAW:
                continue
            to_rotate = first_non_zero(result[i])
            if to_rotate >= 1 and int(get_digit(result[i], to_rotate)) == exclusion:
                nb_raw3 = result[i]
                for j in range(to_rotate, resolution + 1):
                    d3 = int(get_digit(nb_raw3, j))
                    rotated3 = np.uint64((d3 * int(multiplier)) % 7)
                    sh3 = np.uint64(57 - 3 * (j - 1))
                    nb_raw3 = (nb_raw3 & ~(np.uint64(0x07) << sh3)) | (rotated3 << sh3)
                result[i] = nb_raw3

    return result




# ---------------------------------------------------------------------------
# Base-cell neighbour helpers (Python-level — not inner hot path)
# ---------------------------------------------------------------------------

def _is_raw_index(arg) -> bool:
    """True if arg should be treated as a packed uint64 index, not a bare base-cell id.

    Only np.uint64 scalars are raw indices — all other integer types (int, np.uint8, etc.)
    are treated as base-cell IDs and validated against the 0-11 range.
    """
    return isinstance(arg, np.uint64)


def get_base_cell_neighbours(arg) -> list:
    """
    Returns list of 5 neighbouring base cell IDs.
    Accepts either a base-cell int/uint8 (0-11) OR a raw uint64 index at resolution 0.
    """
    if not _is_raw_index(arg):
        bc = int(arg)
        if bc > 11:
            raise ValueError(f"Base cell must be 0-11, got {bc}")
    else:
        raw = np.uint64(arg)
        res = get_resolution(raw)
        if res != 0:
            raise ValueError("This function is for base cells only (resolution 0)")
        bc = int(get_base_cell(raw))
    arr = _BASE_CELL_NEIGHBOURS_RAW[bc].tolist()
    excl = int(_EXCLUDE_NEIGHBOURS[bc])  # 1-based
    return arr[:excl - 1] + arr[excl:]


def get_base_cell_neighbour(arg, direction: int):
    """
    Return the neighbour base-cell ID in direction (0-4).
    Returns None for invalid direction (pentagons have only 5 neighbours).
    """
    if direction < 0 or direction >= 5:
        return None
    if not _is_raw_index(arg):
        bc = int(arg)
        if bc > 11:
            raise ValueError(f"Base cell must be 0-11, got {bc}")
    else:
        bc = int(get_base_cell(np.uint64(arg)))
        if bc > 11:
            raise ValueError(f"Base cell must be 0-11, got {bc}")
    arr = _BASE_CELL_NEIGHBOURS_RAW[bc].tolist()
    excl = int(_EXCLUDE_NEIGHBOURS[bc])
    neighbours = arr[:excl - 1] + arr[excl:]
    return np.uint8(neighbours[direction])


# ---------------------------------------------------------------------------
# Hex / string conversion (Python-level, not performance-critical)
# ---------------------------------------------------------------------------

def decode_z7hex_index(z7_hex_str: str):
    """
    Decode a 16-char hex string → (base_cell: int, resolution_digits: list[int]).
    """
    value = np.uint64(int(z7_hex_str, 16))
    binary = format(int(value), "064b")
    base_cell = int(binary[:4], 2)
    resolution_digits = []
    for i in range(20):
        start = 4 + i * 3
        resolution_digits.append(int(binary[start:start + 3], 2))
    return base_cell, resolution_digits


def encode_z7hex_index(base_cell: int, resolution_digits: list) -> str:
    """
    Encode base cell + resolution digits → 16-char lowercase hex string.
    """
    binary = format(base_cell, "04b")
    padded = list(resolution_digits[:20]) + [7] * max(0, 20 - len(resolution_digits))
    for d in padded:
        binary += format(d, "03b")
    value = int(binary, 2)
    return format(value, "016x")


def z7hex_to_z7string(z7_hex_str: str) -> str:
    base_cell, digits = decode_z7hex_index(z7_hex_str)
    parts = [f"{base_cell:02d}"]
    for d in digits:
        if d == 7:
            break
        parts.append(str(d))
    return "".join(parts)


def z7hex_to_z7int(z7_hex_str: str) -> np.uint64:
    return np.uint64(int(z7_hex_str, 16))


def z7int_to_z7hex(z7_int) -> str:
    return format(int(np.uint64(z7_int)), "016x")


def get_z7hex_resolution(z7_hex_str: str) -> int:
    _, digits = decode_z7hex_index(z7_hex_str)
    return sum(1 for d in digits if 0 <= d < 7)


def get_z7hex_local_pos(z7_hex_str: str):
    z7_string = z7hex_to_z7string(z7_hex_str)
    parent = z7_string[:-1]
    local_pos = z7_string[-1]
    is_center = local_pos == "0"
    return parent, local_pos, is_center


def get_z7string_resolution(z7_string: str) -> int:
    return len(z7_string) - 2


def get_z7string_local_pos(z7_string: str):
    parent = z7_string[:-1]
    local_pos = z7_string[-1]
    is_center = local_pos == "0"
    return parent, local_pos, is_center


def z7string_to_index(z7_string: str) -> np.uint64:
    base_cell = np.uint8(int(z7_string[:2]))
    digits = np.array([int(c) for c in z7_string[2:]], dtype=np.uint8)
    return np.uint64(_encode_z7int_jit(base_cell, digits))


def index_to_z7string(raw) -> str:
    raw = np.uint64(raw)
    bc = int(get_base_cell(raw))
    result = f"{bc:02d}"
    for i in range(1, 21):
        d = int(get_digit(raw, i))
        if d == 7:
            break
        result += str(d)
    return result
