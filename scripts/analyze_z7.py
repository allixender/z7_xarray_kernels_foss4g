import sys
import os
sys.path.insert(0, ".")
import z7py as Z7
import numpy as np

parent_str = "0330"
print(f"Parent: {parent_str}")

# Gather 49 children at res 4
rows = []
for d3 in range(7):
    for d4 in range(7):
        s = parent_str + str(d3) + str(d4)
        idx = Z7.z7string_to_index(s)
        mon = Z7.z7_to_monotonic_int(idx, 4)
        rows.append({"s": s, "idx": idx, "mon": mon, "d3": d3, "d4": d4})

# Analyze patterns
header = f"{'String':<10} | {'Monotonic':<10} | {'Raw Hex':<18} | {'Pure Integer':<25} | {'Binary Digits (3-bit chunks)'}"
print(header)
print("-" * len(header))
for r in rows:
    idx = r["idx"]
    # Digit 1: 59-57
    # Digit 2: 56-54
    # Digit 3: 53-51
    # Digit 4: 50-48
    top_bits = format(idx >> 48, "016b")
    # Group as 4 bits (base cell), then four 3-bit groups
    grouped = f"{top_bits[0:4]} {top_bits[4:7]} {top_bits[7:10]} {top_bits[10:13]} {top_bits[13:16]}"
    print(f"{r['s']:<10} | {r['mon']:<10} | {hex(idx):<18} | {int(idx):<25,} | {grouped} ...")

# Look at the jumps
print("\nIncrements in Raw ID:")
for i in range(len(rows) - 1):
    diff = rows[i+1]["idx"] - rows[i]["idx"]
    if rows[i+1]["d3"] != rows[i]["d3"]:
        # This is a major jump (incrementing digit 3)
        print(f"Jump {rows[i]['s']} -> {rows[i+1]['s']}: {diff} (hex: {hex(diff)})")
    else:
        # Minor increment (incrementing digit 4)
        # Should be constant
        pass

# Check the constant step for digit 4
step_d4 = rows[1]["idx"] - rows[0]["idx"]
print(f"Standard step for digit 4: {step_d4} (hex: {hex(step_d4)})")
# This step corresponds to 1 shifted to the position of digit 4 (bit 48)
print(f"1 << 48 = {1 << 48} (hex: {hex(1 << 48)})")

def show_total_range(res):
    print(f"\nTotal Address Space for Resolution {res}:")
    print("-" * 40)
    
    first_s = "00" + "0" * res
    last_s = "11" + "6" * res
    
    first_idx = Z7.z7string_to_index(first_s)
    last_idx = Z7.z7string_to_index(last_s)
    
    first_mon = Z7.z7_to_monotonic_int(first_idx, res)
    last_mon = Z7.z7_to_monotonic_int(last_idx, res)
    
    # +1 because it is 0-indexed
    total_monotonic_slots = last_mon - first_mon + 1
    actual_cells = Z7.RESOLUTION_STATS[res]["num_cells"]
    
    print(f"First Cell: {first_s:<10} | Monotonic: {first_mon}")
    print(f"Last Cell:  {last_s:<10} | Monotonic: {last_mon}")
    print(f"Total Monotonic Slots: {total_monotonic_slots:,}")
    print(f"Actual Grid Cells:     {actual_cells:,}")
    print(f"Difference (Holes):    {total_monotonic_slots - actual_cells:,}")
    print("\nNote: The 'Holes' exist because pentagons (the 12 base cell centers) ")
    print("have one less child branch than hexagons in the Z7 hierarchy.")

show_total_range(4)
