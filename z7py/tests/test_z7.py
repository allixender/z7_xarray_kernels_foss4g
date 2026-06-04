"""
Tests for the Z7 Python/Numba module.
Mirrors the reference test suite in z7jl/tests_z7.jl.
"""

import math
import numpy as np
import pytest

import z7py as Z7
from z7py.z7 import (
    _INVALID_RAW,
    get_neighbour,
    first_non_zero,
    neighbour_addition_cw,
    neighbour_addition_ccw,
    neighbour_addition_ccw_mod,
)


# ---------------------------------------------------------------------------
# Resolution statistics
# ---------------------------------------------------------------------------

class TestResolutionStats:
    def test_get_resolution_stats_valid(self):
        s0 = Z7.get_resolution_stats(0)
        assert s0["num_cells"] == 12
        assert math.isclose(s0["cls_km"], 8199.5003701)

        s10 = Z7.get_resolution_stats(10)
        assert s10["num_cells"] == 2824752492
        assert math.isclose(s10["area_m2"], 180570.0)

    def test_get_resolution_stats_invalid(self):
        with pytest.raises((ValueError, KeyError)):
            Z7.get_resolution_stats(21)
        with pytest.raises((ValueError, KeyError)):
            Z7.get_resolution_stats(-1)

    def test_get_num_cells(self):
        assert Z7.get_num_cells(0) == 12
        assert Z7.get_num_cells(6) == 1176492
        assert Z7.get_num_cells(20) == 797922662976120012

    def test_get_cell_area(self):
        assert math.isclose(Z7.get_cell_area_m2(0),  51006562172408.9)
        assert math.isclose(Z7.get_cell_area_km2(0), 51006562.1724089)
        assert math.isclose(Z7.get_cell_area_m2(14), 75.2)

    def test_get_cls(self):
        assert math.isclose(Z7.get_cls_m(9),  1268.6064)
        assert math.isclose(Z7.get_cls_km(9), 1.2686064)

    def test_find_resolution_by_value_closest(self):
        res = Z7.find_resolution_by_value(1000.0, "cls_m")
        assert res == 9
        assert math.isclose(Z7.get_cls_m(res), 1268.6064)

    def test_find_resolution_by_value_larger(self):
        res = Z7.find_resolution_by_value(1_000_000, "num_cells", prefer="larger")
        assert res == 6
        assert Z7.get_num_cells(res) >= 1_000_000

    def test_find_resolution_by_value_area_closest(self):
        res = Z7.find_resolution_by_value(100.0, "area_m2")
        assert res == 14
        assert math.isclose(Z7.get_cell_area_m2(res), 75.2)

    def test_find_resolution_by_value_smaller(self):
        res = Z7.find_resolution_by_value(1_000_000, "num_cells", prefer="smaller")
        assert res == 5
        assert Z7.get_num_cells(res) <= 1_000_000

    def test_find_resolution_by_value_invalid_metric(self):
        with pytest.raises((ValueError, KeyError)):
            Z7.find_resolution_by_value(100.0, "invalid_metric")

    def test_find_resolution_by_value_invalid_prefer(self):
        with pytest.raises(ValueError):
            Z7.find_resolution_by_value(100.0, "cls_m", prefer="invalid")

    def test_find_resolution_by_cls_m(self):
        assert Z7.find_resolution_by_cls_m(1000.0) == 9
        res = Z7.find_resolution_by_cls_m(500.0, prefer="larger")
        assert res == 9
        assert Z7.get_cls_m(res) >= 500.0

    def test_find_resolution_by_area_m2(self):
        assert Z7.find_resolution_by_area_m2(100.0) == 14

    def test_find_resolution_by_num_cells(self):
        assert Z7.find_resolution_by_num_cells(1_000_000) == 6


# ---------------------------------------------------------------------------
# Hex / string / int conversion
# ---------------------------------------------------------------------------

Z7_HEX_ID    = "004291d4c313ffff"
Z7_STRING_ID = "0001024435230304"


class TestHexStringConversion:
    def test_decode_z7hex_index(self):
        base_cell, digits = Z7.decode_z7hex_index(Z7_HEX_ID)
        assert base_cell == 0
        assert len(digits) == 20
        assert digits[:14] == [0, 1, 0, 2, 4, 4, 3, 5, 2, 3, 0, 3, 0, 4]
        assert all(d == 7 for d in digits[14:])

    def test_encode_z7hex_index_roundtrip(self):
        base_cell = 0
        digits = [0, 1, 0, 2, 4, 4, 3, 5, 2, 3, 0, 3, 0, 4]
        encoded = Z7.encode_z7hex_index(base_cell, digits)
        assert isinstance(encoded, str)
        assert len(encoded) == 16

        dec_base, dec_digits = Z7.decode_z7hex_index(encoded)
        assert dec_base == base_cell
        assert dec_digits[:len(digits)] == digits

        # full round-trip with original hex
        orig_base, orig_digits = Z7.decode_z7hex_index(Z7_HEX_ID)
        reconstructed = Z7.encode_z7hex_index(orig_base, orig_digits)
        assert reconstructed == Z7_HEX_ID

    def test_z7hex_to_z7string(self):
        z7s = Z7.z7hex_to_z7string("004291D4C313FFFF")
        assert z7s == Z7_STRING_ID
        assert isinstance(z7s, str)

    def test_z7hex_to_z7int(self):
        z7i = Z7.z7hex_to_z7int(Z7_HEX_ID)
        assert z7i == np.uint64(18737691454865407)

    def test_get_z7hex_resolution(self):
        assert Z7.get_z7hex_resolution("004291D4C313FFFF") == 14
        assert Z7.get_z7hex_resolution(Z7.encode_z7hex_index(0, [])) == 0
        assert Z7.get_z7hex_resolution(Z7.encode_z7hex_index(5, [1, 2, 3])) == 3

    def test_get_z7hex_local_pos(self):
        parent, local_pos, is_center = Z7.get_z7hex_local_pos("004291D4C313FFFF")
        assert parent == "000102443523030"
        assert local_pos == "4"
        assert is_center is False

        center_hex = Z7.encode_z7hex_index(0, [1, 2, 3, 0])
        _, local_pos_c, is_center_c = Z7.get_z7hex_local_pos(center_hex)
        assert is_center_c is True
        assert local_pos_c == "0"

    def test_z7int_to_z7hex(self):
        z7i = np.uint64(18737691454865407)
        hex_str = Z7.z7int_to_z7hex(z7i)
        assert hex_str == Z7_HEX_ID
        assert len(hex_str) == 16

    def test_decode_z7int(self):
        z7i = np.uint64(18737691454865407)
        base_cell, digits = Z7.decode_z7int(z7i)
        assert int(base_cell) == 0
        assert len(digits) == 20

    def test_encode_z7int(self):
        enc1 = Z7.encode_z7int(np.uint8(5), np.array([0, 1, 2, 3], dtype=np.uint8))
        assert enc1.dtype == np.uint64

        enc2 = Z7.encode_z7int(np.uint8(5), np.array([0, 1, 2, 3], dtype=np.uint8))
        assert enc1 == enc2

        # round-trip
        base_cell = np.uint8(3)
        digits = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
        encoded = Z7.encode_z7int(base_cell, digits)
        dec_base, dec_digits = Z7.decode_z7int(encoded)
        assert int(dec_base) == int(base_cell)
        assert list(dec_digits[:len(digits)]) == list(digits)

    def test_get_z7string_resolution(self):
        assert Z7.get_z7string_resolution(Z7_STRING_ID) == 14
        assert Z7.get_z7string_resolution("00") == 0
        assert Z7.get_z7string_resolution("001234") == 4

    def test_get_z7string_local_pos(self):
        parent, local_pos, is_center = Z7.get_z7string_local_pos(Z7_STRING_ID)
        assert parent == "000102443523030"
        assert local_pos == "4"
        assert is_center is False

        # consistency with hex version
        ph, lph, ich = Z7.get_z7hex_local_pos("004291D4C313FFFF")
        assert parent == ph
        assert local_pos == lph
        assert is_center == ich


# ---------------------------------------------------------------------------
# Index type operations (raw uint64)
# ---------------------------------------------------------------------------

class TestIndexOps:
    def setup_method(self):
        self.test_digits = np.array([0, 1, 0, 2, 4, 4, 3, 5, 2, 3, 0, 3, 0, 4], dtype=np.uint8)
        self.test_raw = Z7.encode_z7int(np.uint8(0), self.test_digits)

    def test_get_base_cell(self):
        assert int(Z7.get_base_cell(self.test_raw)) == 0
        raw2 = Z7.encode_z7int(np.uint8(5), np.array([1, 2, 3], dtype=np.uint8))
        assert int(Z7.get_base_cell(raw2)) == 5

    def test_get_digit(self):
        assert int(Z7.get_digit(self.test_raw, 1)) == 0
        assert int(Z7.get_digit(self.test_raw, 2)) == 1
        assert int(Z7.get_digit(self.test_raw, 3)) == 0
        assert int(Z7.get_digit(self.test_raw, 4)) == 2
        assert int(Z7.get_digit(self.test_raw, 5)) == 4

    def test_get_digits(self):
        digits = Z7.get_digits(self.test_raw)
        assert len(digits) == 20
        assert list(digits[:14]) == list(self.test_digits)
        assert all(d == 7 for d in digits[14:])

    def test_get_resolution(self):
        assert Z7.get_resolution(self.test_raw) == 14
        raw0 = Z7.encode_z7int(np.uint8(3), np.empty(0, dtype=np.uint8))
        assert Z7.get_resolution(raw0) == 0
        raw3 = Z7.encode_z7int(np.uint8(1), np.array([1, 2, 3], dtype=np.uint8))
        assert Z7.get_resolution(raw3) == 3

    def test_get_parent(self):
        raw = self.test_raw
        assert Z7.get_resolution(raw) == 14

        # one level up
        parent = Z7.get_parent(raw)
        assert Z7.get_resolution(parent) == 13
        assert int(Z7.get_base_cell(parent)) == int(Z7.get_base_cell(raw))
        assert list(Z7.get_digits(parent)[:13]) == list(Z7.get_digits(raw)[:13])
        assert all(d == 7 for d in Z7.get_digits(parent)[13:])

        # explicit resolution
        parent10 = Z7.get_parent(raw, resolution=10)
        assert Z7.get_resolution(parent10) == 10


class TestMonotonicInt:
    def test_round_trip(self):
        # Base cell 0
        raw0 = Z7.z7string_to_index("00")
        m0 = Z7.z7_to_monotonic_int(raw0, 0)
        assert m0 == 0
        assert Z7.monotonic_int_to_z7(m0, 0) == raw0

        # Base cell 5, resolution 14
        raw14 = Z7.z7string_to_index("05012340123401")
        m14 = Z7.z7_to_monotonic_int(raw14, 12)
        assert Z7.monotonic_int_to_z7(m14, 12) == raw14

        # Cross-check kontiguity
        # 050 -> 051 should be +1 in monotonic
        r50 = Z7.z7string_to_index("050")
        r51 = Z7.z7string_to_index("051")
        m50 = Z7.z7_to_monotonic_int(r50, 1)
        m51 = Z7.z7_to_monotonic_int(r51, 1)
        assert m51 == m50 + 1


# ---------------------------------------------------------------------------
# String conversion helpers
# ---------------------------------------------------------------------------

class TestStringConversion:
    def test_z7string_to_index(self):
        idx = Z7.z7string_to_index("0800433")
        assert int(Z7.get_base_cell(idx)) == 0x08
        assert Z7.get_resolution(idx) == 5
        assert int(Z7.get_digit(idx, 1)) == 0
        assert int(Z7.get_digit(idx, 2)) == 0
        assert int(Z7.get_digit(idx, 3)) == 4
        assert int(Z7.get_digit(idx, 4)) == 3
        assert int(Z7.get_digit(idx, 5)) == 3

        idx0 = Z7.z7string_to_index("00")
        assert int(Z7.get_base_cell(idx0)) == 0
        assert Z7.get_resolution(idx0) == 0

        idx11 = Z7.z7string_to_index("1111")
        assert int(Z7.get_base_cell(idx11)) == 0x0B
        assert Z7.get_resolution(idx11) == 2

    def test_index_to_z7string_roundtrip(self):
        for s in ("0800433", "091201", "05"):
            idx = Z7.z7string_to_index(s)
            assert Z7.index_to_z7string(idx) == s


# ---------------------------------------------------------------------------
# first_non_zero
# ---------------------------------------------------------------------------

class TestFirstNonZero:
    def test_cases(self):
        assert first_non_zero(Z7.z7string_to_index("0000000")) == 6
        assert first_non_zero(Z7.z7string_to_index("1000000")) == 6
        assert first_non_zero(Z7.z7string_to_index("1234000")) == 1
        assert first_non_zero(Z7.z7string_to_index("1200567")) == 3
        # resolution 0 — digit 1 is 7 → returns 0
        assert first_non_zero(Z7.z7string_to_index("12")) == 0


# ---------------------------------------------------------------------------
# GBT addition tables
# ---------------------------------------------------------------------------

class TestGBTAddition:
    def test_neighbour_addition_cw(self):
        assert neighbour_addition_cw(np.uint8(0), np.uint8(0)) == (0, 0)
        assert neighbour_addition_cw(np.uint8(0), np.uint8(1)) == (0, 1)
        assert neighbour_addition_cw(np.uint8(1), np.uint8(1)) == (1, 4)
        assert neighbour_addition_cw(np.uint8(1), np.uint8(2)) == (0, 3)

    def test_neighbour_addition_ccw(self):
        assert neighbour_addition_ccw(np.uint8(0), np.uint8(0)) == (0, 0)
        assert neighbour_addition_ccw(np.uint8(0), np.uint8(1)) == (0, 1)
        assert neighbour_addition_ccw(np.uint8(1), np.uint8(1)) == (1, 2)
        assert neighbour_addition_ccw(np.uint8(1), np.uint8(2)) == (0, 3)

    def test_neighbour_addition_ccw_mod(self):
        assert neighbour_addition_ccw_mod(np.uint8(0), np.uint8(0)) == (0, 0)
        assert neighbour_addition_ccw_mod(np.uint8(1), np.uint8(1)) == (1, 2)
        assert neighbour_addition_ccw_mod(np.uint8(6), np.uint8(1)) == (0, 0)


# ---------------------------------------------------------------------------
# Base-cell neighbours
# ---------------------------------------------------------------------------

class TestBaseCellNeighbours:
    def test_get_base_cell_neighbours_bc0(self):
        nbs = Z7.get_base_cell_neighbours(np.uint8(0))
        assert len(nbs) == 5
        assert nbs == [5, 4, 2, 1, 3]

    def test_get_base_cell_neighbours_bc1(self):
        nbs = Z7.get_base_cell_neighbours(np.uint8(1))
        assert len(nbs) == 5
        assert nbs == [5, 0, 6, 10, 2]

    def test_get_base_cell_neighbours_bc6(self):
        nbs = Z7.get_base_cell_neighbours(np.uint8(6))
        assert len(nbs) == 5
        assert nbs == [10, 2, 1, 11, 7]

    def test_get_base_cell_neighbours_bc11(self):
        nbs = Z7.get_base_cell_neighbours(np.uint8(11))
        assert len(nbs) == 5
        assert nbs == [9, 6, 10, 8, 7]

    def test_get_base_cell_neighbours_invalid(self):
        with pytest.raises((ValueError, Exception)):
            Z7.get_base_cell_neighbours(np.uint8(12))

    def test_get_base_cell_neighbour_directions_bc1(self):
        assert int(Z7.get_base_cell_neighbour(np.uint8(1), 0)) == 5
        assert int(Z7.get_base_cell_neighbour(np.uint8(1), 1)) == 0
        assert int(Z7.get_base_cell_neighbour(np.uint8(1), 2)) == 6
        assert int(Z7.get_base_cell_neighbour(np.uint8(1), 3)) == 10
        assert int(Z7.get_base_cell_neighbour(np.uint8(1), 4)) == 2

    def test_get_base_cell_neighbour_invalid_directions(self):
        assert Z7.get_base_cell_neighbour(np.uint8(1), 5) is None
        assert Z7.get_base_cell_neighbour(np.uint8(1), 6) is None
        assert Z7.get_base_cell_neighbour(np.uint8(1), -1) is None

    def test_get_base_cell_neighbour_bc0(self):
        assert int(Z7.get_base_cell_neighbour(np.uint8(0), 0)) == 5
        assert int(Z7.get_base_cell_neighbour(np.uint8(0), 1)) == 4
        assert int(Z7.get_base_cell_neighbour(np.uint8(0), 4)) == 3

    def test_get_base_cell_neighbour_invalid_bc(self):
        with pytest.raises((ValueError, Exception)):
            Z7.get_base_cell_neighbour(np.uint8(12), 0)

    def test_all_base_cells_have_5_neighbours(self):
        for bc in range(12):
            nbs = Z7.get_base_cell_neighbours(np.uint8(bc))
            assert len(nbs) == 5
            assert all(0 <= n <= 11 for n in nbs)
            assert len(set(nbs)) == 5  # no duplicates

    def test_get_base_cell_neighbours_from_index_bc0(self):
        raw = Z7.encode_z7int(np.uint8(0), np.empty(0, dtype=np.uint8))
        nbs = Z7.get_base_cell_neighbours(raw)
        assert nbs == [5, 4, 2, 1, 3]

    def test_get_base_cell_neighbours_from_index_bc1(self):
        raw = Z7.encode_z7int(np.uint8(1), np.empty(0, dtype=np.uint8))
        nbs = Z7.get_base_cell_neighbours(raw)
        assert nbs == [5, 0, 6, 10, 2]

    def test_get_base_cell_neighbours_from_index_non_zero_res_raises(self):
        raw = Z7.encode_z7int(np.uint8(1), np.array([1], dtype=np.uint8))
        with pytest.raises((ValueError, Exception)):
            Z7.get_base_cell_neighbours(raw)

    def test_get_base_cell_neighbours_from_index_bc5(self):
        raw = Z7.encode_z7int(np.uint8(5), np.empty(0, dtype=np.uint8))
        nbs = Z7.get_base_cell_neighbours(raw)
        assert nbs == [4, 0, 10, 9, 1]

    def test_get_base_cell_neighbour_from_index(self):
        raw1 = Z7.encode_z7int(np.uint8(1), np.empty(0, dtype=np.uint8))
        assert int(Z7.get_base_cell_neighbour(raw1, 0)) == 5
        assert int(Z7.get_base_cell_neighbour(raw1, 2)) == 6
        assert Z7.get_base_cell_neighbour(raw1, 5) is None

        raw2 = Z7.encode_z7int(np.uint8(2), np.empty(0, dtype=np.uint8))
        assert int(Z7.get_base_cell_neighbour(raw2, 0)) == 1
        assert int(Z7.get_base_cell_neighbour(raw2, 1)) == 0
        assert Z7.get_base_cell_neighbour(raw2, 6) is None


# ---------------------------------------------------------------------------
# get_neighbour (single direction)
# ---------------------------------------------------------------------------

class TestGetNeighbourSingle:
    def test_direction_3_from_0103(self):
        idx = Z7.z7string_to_index("0103")
        nb_raw, carry = get_neighbour(idx, np.uint8(3), 2)
        assert Z7.index_to_z7string(nb_raw) == "0136"
        assert carry == 0

    def test_direction_5_from_0103(self):
        idx = Z7.z7string_to_index("0103")
        nb_raw, carry = get_neighbour(idx, np.uint8(5), 2)
        assert Z7.index_to_z7string(nb_raw) == "0101"
        assert carry == 0

    def test_multi_level_carry_from_0166(self):
        idx = Z7.z7string_to_index("0166")
        nb_raw, carry = get_neighbour(idx, np.uint8(6), 2)
        assert carry == 6
        assert int(Z7.get_digit(nb_raw, 1)) == 3
        assert int(Z7.get_digit(nb_raw, 2)) == 5


# ---------------------------------------------------------------------------
# get_neighbours (all 6)
# ---------------------------------------------------------------------------

def _test_nbs(ref_str, expected_strs):
    idx = Z7.z7string_to_index(ref_str)
    nbs = Z7.get_neighbours(idx)
    valid = [Z7.index_to_z7string(n) for n in nbs if n != _INVALID_RAW]
    assert set(valid) == set(expected_strs), \
        f"ref={ref_str}: got {sorted(valid)}, expected {sorted(expected_strs)}"


class TestGetNeighboursLevel2:
    def test_0000(self):
        _test_nbs("0000", ["0004", "0006", "0003", "0001", "0005"])

    def test_0100(self):
        _test_nbs("0100", ["0104", "0106", "0103", "0101", "0105"])

    def test_0103(self):
        _test_nbs("0103", ["0106", "0161", "0136", "0134", "0101", "0100"])

    def test_0136(self):
        _test_nbs("0136", ["0161", "0163", "0132", "0130", "0134", "0103"])

    def test_0132(self):
        _test_nbs("0132", ["0163", "0055", "0051", "0133", "0130", "0136"])


# ---------------------------------------------------------------------------
# Pentagon center exclusion
# ---------------------------------------------------------------------------

class TestPentagonExclusion:
    def test_pentagon_center_000(self):
        idx = Z7.z7string_to_index("000")
        nbs = Z7.get_neighbours(idx)

        # exclusion for base cell 0 is direction index 2 (1-based) → nbs[1] invalid
        assert nbs[1] == _INVALID_RAW

        valid = [n for n in nbs if n != _INVALID_RAW]
        assert len(valid) == 5

        expected = {"001", "003", "004", "005", "006"}
        got = {Z7.index_to_z7string(n) for n in valid}
        assert got == expected
