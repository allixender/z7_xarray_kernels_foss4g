# FOSS4G Europe conference Z7 CPI/GBT indexing and neighbourhood

## Title

Efficient Neighbourhood Computation and Cloud-Native Storage for the IGEO7 DGGS Using the Z7 GBT Indexing

https://talks.osgeo.org/foss4g-europe-2026-academic-track/me/submissions/DJFPM8/

## Authors

- Alexander Kmoch; Landscape Geoinformatics Lab, University of Tartu, Estonia
- Weston Renoud; QPS, The Netherlands
- Wai Tik Chan; Landscape Geoinformatics Lab, University of Tartu, Estonia
- Javier Jimenez Shaw; Pix4D, Germany
- Kajetan Marcin Chrapkiewicz; Landscape Geoinformatics Lab, University of Tartu, Estonia
- Evelyn Uuemaa; Landscape Geoinformatics Lab, University of Tartu, Estonia

## keywords

DGGS, Z7, GBT, hexagonal grid, neighbourhood computation, Zarr

## Background

Supplement for the FOSS4G Europe conference academic article: the topic is the Z7 CPI/GBT indexing and neighbourhood calculation use, with practical consolidations including the int64 memory layout, storage layout (e.g. Zarr)  based on ordered parent-based ranges and Python/numba implementation with Xarray/xdggs. 

- Z7 CPI/GBT basics
- XDGGS/Zarr Monotonic RangeIndex and storage layout
- one compute example algorithm (slope)


## Extended abstract

Discrete Global Grid Systems (DGGS) are increasingly adopted as a unified spatial reference framework for organising and analysing multi-source geospatial data at global scale. Among hexagonal DGGS configurations [9], refinement ratio 7 (aperture 7) systems preserve hexagonal symmetry across refinement levels and produce unambiguous indexing hierarchies in which each cell maps to exactly one parent [4]. The IGEO7 system and its Z7 hierarchical integer indexing scheme [1] provide a pure aperture 7 equal-area hexagonal DGGS implemented in the open-source DGGRID software, offering the indexing capabilities popularised by H3 but with true equal-area cells rather than H3's ±50 % cell-area variation. Translating this theoretical advantage into efficient analytical workflows remains challenging [6]: tooling for neighbourhood computation, hierarchy traversal, and array-based interoperation is still maturing, and end-to-end demonstrations entirely in the DGGS index space remain rare. Our research question is whether a vectorised index-arithmetic neighbourhood kernel for Z7, combined with a parent-aligned cloud-native storage layout, can support DGGS-native focal analytics at scale, directly on bit-packed integer indices. We address this through (1) Z7 neighbourhood computation using Generalised Balanced Ternary (GBT) arithmetic on the int64 bit-packed index, (2) alignment of Z7's hierarchical index structure with cloud-native Zarr storage via a monotonic parent-based RangeIndex, and (3) a slope-gradient computation as a representative focal operation entirely in the Z7 index space.
A Z7 index is a 64-bit unsigned integer where the first 4 bits encode the base cell number (0–11, corresponding to the 12 pentagonal cells at the icosahedron vertices), and the remaining 60 bits encode up to 20 resolution digits at 3 bits each (values 0–6, with the sentinel 7 marking digits beyond the cell's resolution). This compact representation enables hierarchical operations through bitwise manipulation: parent extraction reduces to masking and setting trailing digit groups to 7, and resolution determination reduces to scanning for the first sentinel 7. Crucially, all children of a given parent share a common bit prefix, a property that is fundamental to both neighbourhood computation and storage optimisation.
Neighbourhood finding in Z7 leverages GBT arithmetic, a generalisation of balanced ternary to the three axes of a hexagonal grid [3]. Each of the six neighbours is computed by digit-wise addition of a direction vector (digits 1–6), implemented via bitshifting and starting at the finest resolution digit, with carries propagating toward coarser levels. Because aperture 7 alternates orientation by approximately ±19.1° between successive resolutions, the addition tables alternate between clockwise and counter-clockwise variants at odd and even resolutions respectively. Each per-digit operation is a 7×7 table lookup yielding both a result digit and a carry digit, giving overall O(r) complexity for resolution r. When the carry propagates beyond the first resolution digit, the neighbour crosses into an adjacent base cell. This requires a lookup in the icosahedral adjacency table and a possible rotational correction at polar base cells. The 12 pentagonal base cells have only five neighbours and are handled through an exclusion-zone lookup. We present a Python/Numba implementation operating directly on numpy.uint64 arrays of Z7 indices, achieving vectorised batch neighbourhood computation suitable for large-scale array workloads.
The hierarchical prefix property of Z7 directly enables an efficient storage layout for cloud-native Zarr [9] archives. When Z7 indices at a given resolution are sorted numerically, all descendants of a common parent occupy a contiguous integer range, defining a natural monotonic RangeIndex per parent zone. The xarray-xdggs package [2,8] exploits this property: DGGS-indexed data are encoded as one-dimensional Zarr arrays whose coordinate dimension is the sorted Z7 cell ID, with only start and end IDs persisted as coordinates and intermediate IDs reconstructed on demand from the RangeIndex. Chunk boundaries are aligned with parent zones at resolution N−k (k ∈ {2,3,4,5}, chunk-size targets 10–100 MB), so that drill-down, aggregation, and focal computations traverse contiguous storage blocks and minimise I/O in object-store environments. The approach mirrors the optimisation underlying HEALPix's nested scheme but generalises to aperture 7 hierarchies. Zarr group attributes record grid type, indexing scheme, and refinement level following xdggs conventions, producing self-describing archives that can be leveraged at scale.
As a practical case study we compute slope gradients in the Porijogi catchment (Estonia) using elevation extracted from the MERIT Hydro global hydrography DEM [10], an openly licensed 3 arc-second (~90 m) raster product. The DEM is regridded to IGEO7 cell centroids at refinement levels 10 (~479 m characteristic length) and 12 (~68 m) by nearest-neighbour sampling performed in the source raster CRS, avoiding any pre-emptive reprojection. Slope is then computed entirely in the Z7 index space by adapting the Finite Difference Algorithm of Li, McGrath and Stefanakis [5] to hexagonal aperture 7. For each cell, the six neighbours are retrieved via the GBT kernel, and the slope magnitude is computed from the orthogonal partial derivatives. Per-axis distances use either a fast lookup table (level-4 parent distortion accounting for ISEA geometry) or a geodesic mode (pyproj.Geod.inv) for sub-percent accuracy. All software is FOSS4G with no proprietary components. Grid generation and indexing use DGGRID with dggrid4py. xdggs and xdggs-dggrid4py provide DGGS-aware Xarray [7] integration. Storage is Zarr. Chunked computation uses Xarray and Dask. Geodesic distances come from pyproj, and Numba JIT-compiles the kernels.
Our results show that the combination of bit-packed uint64 Z7 indices, Numba-compiled GBT arithmetic, and parent-aligned Zarr chunking forms a coherent and performant DGGS-native analytical stack. Vectorised neighbour lookup on plain numpy arrays enables batch processing of millions of cells per call, while parent-aligned chunking reduces hierarchical queries to O(1) range slices over contiguous storage blocks, in contrast to id-modulo chunking which fragments parent zones across chunks. The Porijogi slope case yields consistent and visually coherent slope fields at IGEO7 levels 10 and 12, with the finer level resolving more local terrain variability and pentagon cells correctly masked to NaN. The geodesic distance mode differs from the fast lookup mode by approximately 1.86 % over the area of interest, well within the ISEA equal-area shape-distortion budget, confirming that the lookup mode is appropriate for production workflows while geodesic remains available for validation.
Reflecting on these findings, per-digit table lookups combine well with Numba's LLVM compilation and SIMD vectorisation. Z7's hierarchical prefix property allows Zarr chunking to align with the DGGS hierarchy without auxiliary structures. In contrast, H3's CPI43 base does not guarantee sorted child contiguity within a parent, complicating equivalent strategies. Limitations include the alternating ±19.1° rotation in aperture 7, pentagon-handling overhead at all resolutions, and Xarray's eager-indexing assumption, which may not scale to full-globe resolution-15 datasets without lazy or virtualised coordinates. We conclude that the proposed Z7 GBT kernel, monotonic RangeIndex storage layout, and FDA slope demonstration together constitute a practical open-source template for DGGS-native focal analytics on aperture 7 hexagonal grids. Future work includes extended k-ring neighbourhoods through iterative GBT addition, Dask-based distributed focal computation, standardisation of DGGS Zarr conventions in alignment with CF Conventions and OGC API DGGS, and analogous treatments for the Z3 (ISEA3H) and ZORDER (ISEA4H) indices also provided by DGGRID. To support reproducibility, all source code and notebooks producing these results are released under open-source licences at https://github.com/allixender/z7_xarray_kernels_foss4g (Z7 kernels, slope notebooks, and Zarr storage experiments), https://github.com/allixender/dggrid4py (Python bindings to DGGRID), and https://github.com/LandscapeGeoinformatics/xdggs-dggrid4py/ (xdggs–dggrid4py integration plugin); the MERIT Hydro DEM is openly distributed by Yamazaki et al. [10].

## Initial references

[1] Kmoch, A., Sahr, K., Chan, W. T., & Uuemaa, E. (2025). IGEO7: A new hierarchically indexed hexagonal equal-area discrete global grid system. AGILE: GIScience Series, 6(32). 

[2] Kmoch, A., et al (2024) XDGGS: A community-developed Xarray package to support planetary DGGS data cube computations, Int. Arch. Photogramm. Remote Sens. Spatial Inf. Sci., XLVIII-4/W12-2024.

[3] Sahr, K. (2019). Central Place Indexing: Hierarchical Linear Indexing Systems for Mixed-Aperture Hexagonal Discrete Global Grid Systems. Cartographica, 54(1).

[4] Sahr, K. (2011). Hexagonal discrete global grid systems for geospatial computing. Archives of Photogrammetry Cartography and Remote Sensing.

[5] Li, M., McGrath, H., and Stefanakis, E. (2022). Multi-resolution topographic analysis in hexagonal Discrete Global Grid Systems. International Journal of Applied Earth Observation and Geoinformation.

[6] Hojati, Majid, Colin Robertson, Steven Roberts, and Chiranjib Chaudhuri. 2022. “GIScience
research challenges for realizing discrete global grid systems as a Digital Earth.” Big Earth
Data 6 (3): 358–379.

[7] Hoyer, Stephan, and Joe Hamman. 2017. “xarray: N-D labeled Arrays and Datasets in
Python.” JORS 5 (1): 10.

[8] Mahdavi-Amiri, Ali, Erika Harrison, and Faramarz Samavati. 2015. “Hexagonal connectivity
maps for Digital Earth.” International Journal of Digital Earth 8 (9): 750–769.

[9] Miles, Alistair, John Kirkham, Martin Durant, James Bourbeau, Tarik Onalan, Joe Hamman,
Zain Patel, et al. 2020. “zarr-developers/zarr-python: v2.4.0.” Jan. Accessed 2025-12-12.
https://zenodo.org/record/3773450.

[10] Yamazaki, Dai, Daiki Ikeshima, Jeison Sosa, Paul D. Bates, George H. Allen, and Tamlin M.
Pavelsky. 2019. “MERIT Hydro: A High-Resolution Global Hydrography Map Based on
Latest Topography Dataset.” Water Resources Research 55 (6): 5053–5073.
