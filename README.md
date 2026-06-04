# FOSS4G Europe conference Z7 CPI/GBT indexing and neighbourhood

## Title

Efficient Neighbourhood Computation and Cloud-Native Storage for the IGEO7 DGGS Using the Z7 GBT Indexing

https://talks.osgeo.org/foss4g-europe-2026-academic-track/me/submissions/DJFPM8/

## Authors

- Alexander Kmoch; Landscape Geoinformatics Lab, University of Tartu, Estonia; alexander.kmoch@ut.ee;
- Weston Renoud; QPS, The Netherlands; weston.renoud@qps.nl; 
- Wai Tik Chan; Landscape Geoinformatics Lab, University of Tartu, Estonia; wai.tik.chan@ut.ee; 
- Javier Jimenez Shaw; Pix4D, Germany; j1@jimenezshaw.com;
- Kajetan Marcin Chrapkiewicz; Landscape Geoinformatics Lab, University of Tartu, Estonia; kajetan.marcin.chrapkiewicz@ut.ee;
- Evelyn Uuemaa; Landscape Geoinformatics Lab, University of Tartu, Estonia; evelyn.uuemaa@ut.ee

## keywords

DGGS, Z7, GBT, hexagonal grid, neighbourhood computation, Zarr

## Background

We prepare an article draft for the FOSS4G Europe conference academic draft: the topic is the Z7 CPI/GBT indexing and neighbourhood calculation use, with practical consolidations including the int64 memory layout, storage layout (e.g. Zarr)  based on ordered parent-based ranges and Python/numba implementation with Xarray/xdggs. 

- Z7 CPI/GBT basics
- XDGGS/Zarr Monotonic RangeIndex and storage layout
- (at least) one compute example algorithm (slope?)

The paper has the general classic article layout with sections to fill (all just in markdown).

## The extended abstract section of 800-1000 words.

Discrete Global Grid Systems (DGGS) are increasingly adopted as a unified spatial reference framework for organising and analysing multi-source geospatial data at global scale. Among hexagonal DGGS configurations, refinement ratio 7 systems exhibit particularly desirable properties: they preserve hexagonal symmetry across refinement levels and produce unambiguous indexing hierarchies where each cell maps to exactly one parent (Sahr, 2011). The recently introduced IGEO7 system and its associated Z7 hierarchical integer indexing scheme (Kmoch et al., 2025) provide a pure aperture 7 equal-area hexagonal DGGS implemented in the open-source DGGRID software. While IGEO7/Z7 offers significant theoretical advantages over systems such as H3, such as true equal-area cells versus H3’s ±50% cell size variation, practical challenges remain in translating these advantages into efficient, scalable computational workflows. This paper addresses three interconnected challenges: (1) the algorithmic foundations of Z7 neighbourhood computation using Generalised Balanced Ternary (GBT) arithmetic on the int64 bit-packed index, (2) the alignment of Z7’s hierarchical index structure with cloud-native storage layouts in Zarr via monotonic parent-based range indexing, and (3) a practical demonstration through slope gradient computation as a representative focal operation on hexagonal DGGS.

A Z7 index is a 64-bit unsigned integer where the first 4 bits encode the base cell number (0–11, corresponding to the 12 pentagonal cells at the icosahedron vertices), and the remaining 60 bits encode up to 20 resolution digits at 3 bits each (values 0–6, with 7 marking digits beyond the cell’s resolution). This compact bit-packed representation enables efficient hierarchical operations through bitwise manipulation: parent extraction requires only masking and setting trailing digit groups to 7, while resolution determination reduces to scanning for the first occurrence of the sentinel value 7. The encoding ensures that all children of a given parent cell share a common bit prefix, a property that is fundamental to both neighbourhood computation and storage optimisation.

Neighbourhood finding in Z7 leverages GBT arithmetic, which is a generalisation of balanced ternary to the three axes of a hexagonal grid (Sahr, 2019). Each of the six neighbours of a cell is computed by performing digit-wise addition of a direction vector (digits 1–6) to the cell’s index implemented via bitshifting, starting at the finest resolution digit and propagating carries toward coarser levels. Because the aperture 7 grid alternates orientation between successive resolutions (approximately ±19.1° rotation), the addition tables alternate between clockwise and counter-clockwise variants at odd and even resolutions respectively. Each per-digit operation involves a table lookup (a 7×7 matrix yielding both the result digit and a carry digit), making the overall algorithm O(r) in complexity where r is the resolution. When the carry propagates beyond the first resolution digit, the neighbour crosses into an adjacent base cell, requiring a lookup in the icosahedral adjacency table and potential rotational corrections at polar base cells. We present a Python/Numba implementation of this algorithm that operates directly on arrays of uint64 Z7 indices, achieving vectorised batch neighbourhood computation suitable for large-scale raster-style analysis.

The hierarchical prefix property of Z7 indices, where all children of a parent share a common bit prefix, directly enables an efficient storage layout for cloud-native Zarr archives. When Z7 indices at a given resolution are sorted numerically, cells within the same parent region are stored contiguously. This creates a monotonic range index where any parent zone’s children can be retrieved through a simple range query on the sorted 1-D index dimension. We describe how xarray-xdggs (XDGGS, Kmoch et al, 2024) exploits this property: DGGS-indexed data is encoded as 1-D Zarr arrays with the Z7 cell ID as the coordinate dimension (but stored only as the start and end IDs), and chunking boundaries are aligned with coarser-resolution parent boundaries (e.g., chunking at resolution N-4 or N-5 while storing data at resolution N). This alignment ensures that hierarchical queries, i.e. aggregation from index children to logical parents, or drill-down from parents to children, traverse contiguous storage blocks, minimising I/O operations in cloud-object-storage environments. The approach mirrors the storage optimisation patterns known from quad-trees overviews or the HEALPix nested indexing but extends naturally to aperture 7 hierarchies. We also discuss how Zarr’s metadata attributes are used to record DGGS parameters (grid type, indexing scheme, refinement level), enabling self-describing archives that can be leveraged for large scale computations.

As a practical demonstration, we implement a slope gradient computation, a classical focal GIS operation in terrain analysis, operating entirely within the Z7 index space. For each cell, the algorithm retrieves the six neighbours via GBT arithmetic, obtains the elevation values and their relative directions to each other, calculates the elevation differences along the three hexagonal axes onto two orthogonal components, and computes the slope from the combined partial derivatives. This finite-difference approach, adapted from Li et al. (2022), benefits from the uniform adjacency and equal weighting inherent to hexagonal cells, eliminating the directional bias present in rectangular grid slope computations. We implement this using Xarray for data management, Numba-accelerated functions for batch Z7 neighbour lookups on uint64 arrays, and demonstrate the end-to-end workflow from Zarr-stored elevation data through to a Zarr-stored slope product. All data is indexed by Z7 cell IDs without any intermediate coordinate transformations.

Our results show that the combination of Z7’s bit-packed int64 representation, GBT-based neighbourhood arithmetic, and parent-aligned Zarr chunking provides a coherent, performant stack for DGGS-native geospatial analysis. The approach is implemented using open-source Python tools (Numba, Xarray, XDGGS, Zarr) and is designed to integrate with emerging DGGS standards and cloud-native data infrastructure.

## Initial references

Kmoch, A., Sahr, K., Chan, W. T., & Uuemaa, E. (2025). IGEO7: A new hierarchically indexed hexagonal equal-area discrete global grid system. AGILE: GIScience Series, 6(32). 

Kmoch, A., et al (2024) XDGGS: A community-developed Xarray package to support planetary DGGS data cube computations, Int. Arch. Photogramm. Remote Sens. Spatial Inf. Sci., XLVIII-4/W12-2024.

Sahr, K. (2019). Central Place Indexing: Hierarchical Linear Indexing Systems for Mixed-Aperture Hexagonal Discrete Global Grid Systems. Cartographica, 54(1).

Sahr, K. (2011). Hexagonal discrete global grid systems for geospatial computing. Archives of Photogrammetry Cartography and Remote Sensing.

Li, M., McGrath, H., and Stefanakis, E. (2022). Multi-resolution topographic analysis in hexagonal Discrete Global Grid Systems. International Journal of Applied Earth Observation and Geoinformation.

