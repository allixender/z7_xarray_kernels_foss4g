# Z7

This is a work-in-progress to explore the possibility of neighbor traversal in the IGEO7/Z7. Z7 is an indexing system for the IGEO7 aperture 7 hexagonal discrete global grid DGGRID/Sahr Kmoch et al. (2025). It is based on the Generalized Balanced Ternary (GBT) numeral system described in Lucas, Gibson (1982), van Roessel (1988), Sahr (2019), and Wikipedia (2025).

## Acknowledgements

2019-2026 Kevin Sahr <sahrk@sou.edu> DGGRID https://github.com/sahrk/DGGRID/

https://github.com/wrenoud/Z7/

- 2025-2026 Javier Jimenez Shaw https://github.com/jjimenezshaw
- 2025-2026 Weston James Renoud https://github.com/wrenoud


This is a work-in-progress to explore the possibility of neighbor traversal in the Z7 domain. Z7 is an indexing system
for aperature 7 hexagonal discrete global grid systems, as coined in [Kmoch et al. (2025)][kmoch2025]. It is based on
the Generalized Balanced Ternary (GBT) numeral system described in [Lucas, Gibson (1982)][lucas1982],
[van Roessel (1988)][vanRoessel1988], and [Wikipedia (2025)][wikipedia2025].

### Rotation Pattern

Lucas and Gibson (1982) gave examples of GBT with exclusively counter-clockwise (CCW) rotation. The disadvantage of this
approach is that the orientation of the hexagons at the different hierarchy levels quickly diverge from each other.
[White et al. (1992)][white1992] proposed an alternating rotation direction (CW, CCW, CW, ...) to maintain an alignment
of hexagon
orientations across hierarchy levels. Kmoch et al. (2025) adopted this alternating rotation pattern for Z7 assigning CW
to odd resolutions and CCW to even resolutions.

For reference purposes a rendering of three hierarchies of alternating GBT rotation is included (CW, CCW, CW).

<img src="./images/grid.svg" width="200" alt="3 hierarchies in Z7's alternating GBT pattern">

### GBT Addition

Make nicer figures?

https://github.com/wrenoud/Z7/?tab=readme-ov-file

---

## High-Performance GBT Implementation

The `z7py` implementation uses a bit-packed 64-bit representation to enable extremely fast neighbor traversal and range arithmetic.

### Bit Layout
Indices are stored as `uint64` with the following structure:
- **Bits 63–60:** Base Cell ID (0–11).
- **Bits 59–0:** 20 refinement digits, each occupying 3 bits.
- **Value 7:** A special padding digit indicating the end of the identifier.

### Neighbor Traversal and Cascading Carries
Neighbor finding is implemented using GBT arithmetic tables (`_GBT_CW_*` and `_GBT_CCW_*`) compiled with Numba.
1. **Alternating Rotations:** The implementation automatically switches between Clockwise (CW) and Counter-Clockwise (CCW) addition tables based on whether the resolution level is odd or even.
2. **Cascading Carries:** Moving across a cell boundary often triggers a "carry" to the parent level. The algorithm recursively ripples these carries up the hierarchy (similar to carrying a 1 in decimal addition) until the carry is 0 or a base-cell boundary is reached.
3. **Bitwise Efficiency:** All digit extractions and updates use raw bit-shifts (`>>`) and masks (`&`), ensuring O(1) performance for single-level operations and avoiding expensive string or list allocations.

---

## Sorted Range Index for Z7 (IGEO7) DGGS

## Overview
The Z7 indexing scheme (IGEO7) represents a hierarchical discrete global grid. When data is spatially sorted using Z7 identifiers, it naturally clusters along space-filling curves. The **Sorted Range Index** is a conceptual optimization for Xarray and Zarr that exploits this clustering to provide O(1) or O(log N) data access without loading massive coordinate arrays. 

**Note:** The primary goal of this index is not necessarily to perform exact spatial filtering (such as polygon intersections, or parent-aligned spatial children), but rather to enable highly efficient access to specific Z7 identifiers or larger blocks by mapping them directly to their physical storage locations, reducing chunk loading efforts for close cells.

## Key Conceptual Pillars

### 1. The Base-7 Transition
While Z7 identifiers are natively stored as packed `uint64` (Base-2 bit patterns), this representation often contains numerical "gaps" between siblings.
- **Base-7 Encoding:** By converting Z7 strings or bit-patterns into Base-7 integers, siblings (children of the same parent) become numerically adjacent (incrementing by 1).
- **Contiguous Ranges:** In Base-7, a parent and all its recursive children form a single, contiguous numerical range. This allows representing millions of points with just two values: `[start, end]`.

### 2. Range Compression & Multi-Range Indexing
A dataset can be represented as a collection of these contiguous ranges. Instead of an index of size $N$ (where $N$ is the number of cells), we maintain an index of size $R$ (where $R$ is the number of contiguous ranges). In well-sorted DGGS datasets, $R \ll N$.

### 3. Zarr & Dask Alignment
The index is designed to be "chunk-aware":
- **Chunk-Bound Ranges:** Ranges are ideally calculated per Zarr chunk using Dask.
- **Efficient Filtering:** When querying a spatial region, the index identifies which ranges overlap the query, mapping them directly to specific Zarr chunks. This bypasses the need to read the `zone_id` coordinate variable from disk for non-relevant chunks.

---

## Conceptual Implementation idea

The general concept of contiguous ranges for DGGS indices:

```python
import pandas as pd

# Convert base-7 strings to integers for easier comparison
def base7_to_int(s):
    return int(s, 7)


# Convert integers back to base-7 strings with proper padding
def int_to_base7(n, length):
    if n == 0:
        return '0' * length
    digits = []
    while n:
        digits.append(str(n % 7))
        n //= 7
    return ''.join(reversed(digits)).zfill(length)


def find_continuous_ranges(z7str_cell_ids):
    """
    Find all continuous ranges in a pandas Series of base-7 digit strings.
    
    Parameters:
    series: pd.Series of strings containing digits 0-6
    
    Returns:
    List of tuples (start, end) representing continuous ranges
    """
    if len(z7str_cell_ids) == 0:
        return []
    
    # Remove duplicates and sort
    unique_sorted = sorted(z7str_cell_ids.unique())
    
    
    # Convert to integers
    int_values = [base7_to_int(x) for x in unique_sorted]
    length = len(unique_sorted[0])
    
    # Find continuous ranges
    ranges = []
    start_idx = 0
    
    for i in range(1, len(int_values)):
        # Check if there's a gap
        if int_values[i] != int_values[i-1] + 1:
            # End of a range
            ranges.append((unique_sorted[start_idx], unique_sorted[i-1]))
            start_idx = i
    
    # Add the last range
    ranges.append((unique_sorted[start_idx], unique_sorted[-1]))
    
    return ranges

# Example usage
data = pd.Series(['00001','00002','00003', '00004', '00012', '00013', '00014', '00030', '00031', '00012'])

ranges = find_continuous_ranges(data)

print(ranges)
# Expected Output: [('00001', '00004'), ('00012', '00014'), ('00030', '00031')]

print(len(ranges))
# Expected Output: 3
```

---

## Performance and Correctness Considerations

- **Sorting Requirement:** The efficiency and correctness of this index rely on the data being lexicographically sorted by Z7 identifier during the dataset creation (regridding) phase. If the data is shuffled, the ranges degrade into individual points.
- **Coordinate Transformation:** The index provides a bidirectional mapping:
    - **Forward:** Maps an absolute position in the data array to a Z7 identifier.
    - **Reverse:** Maps a Z7 identifier (or range) back to absolute positions/slices in the data array.
- **Lazy Evaluation:** By utilizing Dask for range calculation and storage, the index remains lightweight and integrates seamlessly with Xarray's lazy loading mechanisms.
- **Bit-Level Accuracy:** Conversions between Base-2 (packed uint64) and Base-7 must be handled carefully (e.g., using `np.unpackbits` or bitwise shifts) to ensure no loss of precision in the face-ID or refinement levels.

---

## Leveraging High-Performance Kernels

The `RangeIndex` can be significantly optimized by replacing string-based or byte-unpacking logic with the specialized bitwise kernels found in `z7py/z7.py`.

### 1. Performance vs. Memory
The prototype's use of `np.unpackbits` creates an intermediate boolean representation of the dataset, which is memory-intensive. By using Numba-jitted bitwise operators (such as those in `z7py`), we can:
- **Zero Allocations:** Perform conversions directly on the `uint64` stream without creating temporary arrays.
- **CPU Instruction Level Speed:** Bitwise shifts and masks map directly to hardware instructions, providing several orders of magnitude faster execution than Python-level loops or numpy sum-reductions.

### 2. Monotonic Integer Mapping
To ensure that Z7 identifiers are truly contiguous for range arithmetic, we map the hierarchical Z7 structure to a **Monotonic Integer**:
$$Value = (BaseCell \times 7^L) + \sum_{i=1}^L (Digit_i \times 7^{L-i})$$

#### Why Monotonicity is Critical:
In this context, "monotonic" refers to a strict, one-to-one mapping between the hierarchical grid and a continuous sequence of integers. This property is foundational for several reasons:

- **Hierarchy as a Number Line:** Monotonicity ensures that spatial hierarchy translates directly into numerical continuity. If a parent has a monotonic ID of $P$, its children are guaranteed to have perfectly sequential IDs (e.g., $P \times 7 + [0..6]$).
- **Enabling Range Compression:** The efficiency of a `RangeIndex` relies on the ID space being **dense**. If the mapping were not monotonic, a range `[start, end]` might contain "ghost" IDs that don't exist in the grid. Monotonicity ensures every integer in a range corresponds to exactly one valid Z7 cell.
- **Constant-Time Position Calculation ($O(1)$):** Because the IDs are perfectly sequential, we can calculate the physical offset of any cell within a chunk using simple subtraction:
  $$\text{Offset} = \text{TargetID} - \text{StartID}$$
  This bypasses the need for expensive searches or lookups during data retrieval.

#### Implementation Specifics:
By using powers of 7 (the refinement factor of Z7), each refinement level occupies exactly its required "slot" in the number line. Placing the `BaseCell` (0-11) as the most significant component ensures that the index jumps cleanly from one major face of the Earth to the next without overlapping values or numerical "bumps."

### 3. Numba Implementation Choices
In `z7py`, several choices maximize performance:
- **`@njit(cache=True)`:** Compiles the indexing logic to machine code, bypassing the Python Global Interpreter Lock (GIL) and allowing Dask to execute the range calculation across multiple threads efficiently.
- **Lookup Tables in Memory:** Constants like `_BASE_CELL_NEIGHBOURS_RAW` are stored as contiguous numpy arrays, allowing the Numba compiler to optimize memory access patterns during index traversal.
- **Static Typing:** By forcing `np.uint64` and `np.uint8`, we avoid expensive type-checking and boxing/unboxing overhead in the hot path of the RangeIndex creation.

---

## Draft Implementation: `Z7MonotonicIndex`

The following is a draft implementation of the optimized RangeIndex, leveraging `z7py` for bitwise speed and `dask` for scalability.

```python
import numpy as np
import numba as nb
import dask.array as da
import xarray as xr
from z7py import z7_to_monotonic_int, monotonic_int_to_z7

@nb.njit(cache=True)
def get_monotonic_ranges_jit(chunk_uint64, resolution):
    """
    Numba-optimized kernel to find contiguous ranges in a chunk.
    Returns: Array of [length, start_monotonic, end_monotonic]
    """
    n = len(chunk_uint64)
    if n == 0:
        return np.zeros((0, 3), dtype=np.uint64)
    
    # 1. Convert to monotonic integers (O(N) bitwise)
    monotonic = np.empty(n, dtype=np.uint64)
    for i in range(n):
        monotonic[i] = z7_to_monotonic_int(chunk_uint64[i], resolution)
    
    # 2. Identify contiguous segments
    # In a sorted DGGS, most segments will be large blocks
    diffs = np.diff(monotonic)
    gaps = np.where(diffs != 1)[0]
    
    starts = np.concatenate((np.array([0]), gaps + 1))
    ends = np.concatenate((gaps, np.array([n - 1])))
    
    # 3. Build range table
    num_ranges = len(starts)
    result = np.empty((num_ranges, 3), dtype=np.uint64)
    for i in range(num_ranges):
        s_idx, e_idx = starts[i], ends[i]
        result[i, 0] = e_idx - s_idx + 1 # Length
        result[i, 1] = monotonic[s_idx]  # Start ID
        result[i, 2] = monotonic[e_idx]  # End ID
        
    return result

class Z7MonotonicIndex(xr.indexes.Index):
    """
    Xarray Index for Z7 DGGS that uses compressed monotonic ranges.
    Replaces Z7 string/hex identifiers with efficient integer range arithmetic.
    """
    def __init__(self, ranges, resolution, dim="zone_id", coord_name=None):
        # ranges: dask array of [abs_start_pos, start_monotonic_id, end_monotonic_id]
        self.ranges = ranges 
        self.resolution = resolution
        self.dim = dim
        self.coord_name = coord_name or dim

    @classmethod
    def from_z7_array(cls, data_array):
        """Scaffold the index from an existing Xarray DataArray of Z7 uint64."""
        res = data_array.attrs.get("level", 14)
        
        def process_chunk(chunk, block_info=None):
            # Calculate ranges for this specific chunk
            chunk_ranges = get_monotonic_ranges_jit(chunk, res)
            # Offset the local positions to absolute positions
            offset = block_info[0]['array-location'][0][0]
            chunk_ranges[:, 0] += offset 
            return chunk_ranges

        # Lazy calculation of ranges across Zarr chunks
        raw_ranges = data_array.data.map_blocks(
            process_chunk, 
            chunks=(None, 3), 
            dtype=np.uint64
        )
        return cls(raw_ranges, res, dim=data_array.dims[0])

    def sel(self, labels, method=None, tolerance=None):
        """
        The "Reverse Mapping": Z7 ID -> Physical Position
        Uses binary search over ranges + monotonic subtraction (O(log R)).
        """
        target_z7 = labels[self.dim]
        target_m = z7_to_monotonic_int(target_z7, self.resolution)
        
        # 1. Find the range containing target_m (using dask/numpy search)
        # range_idx = search_ranges(self.ranges, target_m)
        
        # 2. Calculate O(1) offset
        # pos = range.abs_start + (target_m - range.start_id)
        
        # ... logic to return IndexSelResult ...
        pass

    def isel(self, indexers):
        """
        The "Forward Mapping": Physical Position -> Z7 ID
        Maps integer slices back to Z7 labels.
        """
        # pos = indexers[self.dim]
        # label_m = ranges.lookup_pos(pos)
        # return monotonic_int_to_z7(label_m, self.resolution)
        pass
```

---

## Development with Pixi

This project uses [Pixi](https://pixi.sh), a modern, high-performance package manager built on the Conda ecosystem. Pixi is particularly well-suited for scientific projects because it provides strict reproducibility through lockfiles and handles complex C/C++ dependencies (like those often found in GIS and numerical libraries) better than standard `pip`.

### Getting Started

1.  **Install Pixi:**
    ```bash
    curl -fsSL https://pixi.sh/install.sh | bash
    ```
    *(For other platforms, see the [installation guide](https://pixi.sh/latest/#installation)).*

2.  **Setup and Run Tests:**
    Pixi automatically manages the environment for you. You don't need to `pip install` anything.
    ```bash
    pixi run test
    ```

3.  **Enter the Development Environment:**
    If you want to run a script or start a REPL within the project environment:
    ```bash
    pixi shell
    python
    ```

### Jupyter Integration

For interactive science and experimentation, you can easily use this Pixi environment with Jupyter:

1.  **Add Jupyter to the project:**
    ```bash
    pixi add jupyterlab ipykernel
    ```

2.  **Run Jupyter Lab directly:**
    ```bash
    pixi run jupyter lab
    ```

3.  **Using with VS Code or external Jupyter:**
    If you prefer using VS Code, simply select the Python interpreter located in `.pixi/envs/default/bin/python`. VS Code will automatically recognize the environment and its packages.

### Why Pixi?
-   **Reproducibility:** The `pixi.lock` file ensures that every collaborator uses the exact same versions of all dependencies, including Python itself and system-level libraries.
-   **No Activation Needed:** Unlike `conda` or `venv`, you don't need to manually activate environments to run tasks. `pixi run <task>` handles it instantly.
-   **Unified Conda & Pip:** Pixi can install and manage dependencies from both Conda channels (like `conda-forge`) and PyPI (`pip`) in the same environment. This solves the "missing package" problem that often plagues tools like `poetry` or `micromamba`.
-   **Project-Local Environments:** Unlike Conda/Micromamba which use global named environments, Pixi stores the environment inside the project directory (`.pixi/`). This makes it much easier to run isolated experiments with different package versions without polluting your system or forgetting which "env" was for which project.
-   **Multi-language:** It handles Python, R, C++, and more, making it ideal for projects that bridge high-level analysis and low-level kernels.


---


## References

1. Kmoch, A., Sahr, K., Chan, W. T., & Uuemaa, E. (2025).
   [IGEO7: A new hierarchically indexed hexagonal equal-area discrete global grid system][kmoch2025]. _AGILE: GIScience
   Series, 6_(32). https://doi.org/10.5194/agile-giss-6-32-2025
2. Lucas, D., & Gibson, L. (1982). [_Automated Analysis of imagery_][lucas1982]. Air Force Office of Scientific
   Research. https://apps.dtic.mil/sti/tr/pdf/ADA125710.pdf. (No. AFOSRTR830055).
3. Sahr, K. (2011). [Hexagonal discrete global grid systems for geospatial computing][sahr2011]. _Archives of
   Photogrammetry Cartography and Remote Sensing, 22_, 363–376.
4. van Roessel, J. W. (1988).
   [Conversion of Cartesian coordinates from and to Generalized Balanced Ternary addresses][vanRoessel1988].
   _Photogrammetric Engineering & Remote Sensing, 54_(11), 1565–1570.
5. White, D., Kimerling, J. A., & Overton, S. W. (1992).
   [Cartographic and Geometric Components of a Global Sampling Design for Environmental Monitoring][white1992].
   _Cartography and Geographic Information Systems_, 19(1), 5-22. https://doi.org/10.1559/152304092783786636
6. Wikipedia Contributors. (2025, December 16). [_Generalized balanced ternary_][wikipedia2025]. Wikipedia; Wikimedia
   Foundation.

---

[wikipedia2025]: https://en.wikipedia.org/wiki/Generalized_balanced_ternary

[kmoch2025]: https://agile-giss.copernicus.org/articles/6/32/2025/agile-giss-6-32-2025.pdf

[vanRoessel1988]: https://web.archive.org/web/20250523045258/https://www.asprs.org/wp-content/uploads/pers/1988journal/nov/1988_nov_1565-1570.pdf

[white1992]: https://www.tandfonline.com/doi/pdf/10.1559/152304092783786636?casa_token=lyaVPE0vjcQAAAAA:GL0QWw01pvr2AQWJgmNcjHH_rCp3VLv5jayMi7O9Kb7jC_l2KFt2b98GvnRcOhi4qSXV-FXXxUYAVg

[sahr2011]: https://www.researchgate.net/publication/266415994_Hexagonal_discrete_global_grid_systems_for_geospatial_computing

[lucas1982]: https://apps.dtic.mil/sti/tr/pdf/ADA125710.pdf