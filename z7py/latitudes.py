"""
Authalic ↔ Geodetic latitude conversion for WGS84 ellipsoid.
Translated from z7jl/AuthalicLatitude.jl.
"""

import numpy as np
import numba as nb

# The order of the series is a global constant
POLYNOMIAL_ORDER = 6

# WGS84 ellipsoid parameters
WGS84_A = 6378137.0              # Semi-major axis (meters)
WGS84_F = 1.0 / 298.257223563    # Flattening

# Geodetic to authalic: Coefficients for converting ϕ to ξ.
# Eq. A19 in Karney (2022)
# These are the same as in the Julia code, 6x6 matrix.
AUTHALIC_FWD = np.array([
    [-4/3, -4/45, 88/315, 538/4725, 20824/467775, -44732/2837835],
    [0, 34/45, 8/105, -2482/14175, -37192/467775, -12467764/212837625],
    [0, 0, -1532/2835, -898/14175, 54968/467775, 100320856/1915538625],
    [0, 0, 0, 6007/14175, 24496/467775, -5884124/70945875],
    [0, 0, 0, 0, -23356/66825, -839792/19348875],
    [0, 0, 0, 0, 0, 570284222/1915538625]
], dtype=np.float64)

# Authalic to geodetic: Coefficients for converting ξ to ϕ.
# Eq. A20 in Karney (2022)
AUTHALIC_INV = np.array([
    [4/3, 4/45, -16/35, -2582/14175, 60136/467775, 28112932/212837625],
    [0, 46/45, 152/945, -11966/14175, -21016/51975, 251310128/638512875],
    [0, 0, 3044/2835, 3802/14175, -94388/66825, -8797648/10945935],
    [0, 0, 0, 6059/4725, 41072/93555, -1472637812/638512875],
    [0, 0, 0, 0, 768272/467775, 455935736/638512875],
    [0, 0, 0, 0, 0, 4210684958/1915538625]
], dtype=np.float64)

# Coefficients for expansion of the normalized meridian arc unit in terms
# of *n²*, the square of the third flattening.
# See [Karney 2010](crate::Bibliography::Kar10) eq. (29)
MERIDIAN_ARC_COEFFICIENTS = np.array([
    1.0,
    1.0/4.0,
    1.0/64.0,
    1.0/256.0,
    25.0/16384.0,
    49.0/65536.0,
    441.0/1048576.0
], dtype=np.float64)

@nb.njit(cache=True)
def horner(arg, coeffs):
    """
    Evaluate a polynomial Σ cᵢ ⋅ xⁱ using the efficient Horner's scheme.
    `coeffs` should be ordered from the lowest power (c₀, c₁, ...) to the highest.
    """
    val = coeffs[-1]
    for i in range(len(coeffs) - 2, -1, -1):
        val = val * arg + coeffs[i]
    return val

@nb.njit(cache=True)
def fourier_sin(angle, coeffs):
    """
    Evaluate a Fourier sine series: Σ cᵢ * sin(i * angle).
    """
    val = 0.0
    for i in range(len(coeffs)):
        val += coeffs[i] * np.sin((i + 1) * angle)
    return val

@nb.njit(cache=True)
def third_flattening(f):
    """
    The third flattening, n = f / (2 - f).
    """
    return f / (2.0 - f)

@nb.njit(cache=True)
def normalized_meridian_arc_unit(f):
    """
    The Normalized Meridian Arc Unit, Qn.
    """
    n = third_flattening(f)
    return horner(n * n, MERIDIAN_ARC_COEFFICIENTS) / (1.0 + n)

@nb.njit(cache=True)
def compute_fourier_coeffs(f, poly_fwd, poly_inv):
    """
    Compute Fourier coefficients by evaluating their corresponding Taylor polynomials.
    """
    n = third_flattening(f)
    fwd = np.empty(POLYNOMIAL_ORDER, dtype=np.float64)
    inv = np.empty(POLYNOMIAL_ORDER, dtype=np.float64)
    for i in range(POLYNOMIAL_ORDER):
        fwd[i] = n * horner(n, poly_fwd[i])
        inv[i] = n * horner(n, poly_inv[i])
    
    # etc[0] in Julia corresponds to normalized_meridian_arc_unit
    qn = normalized_meridian_arc_unit(f)
    return fwd, inv, qn

@nb.njit(cache=True)
def _geodetic_to_authalic_rad(phi_rad, fwd_coeffs):
    return phi_rad + fourier_sin(2.0 * phi_rad, fwd_coeffs)

@nb.njit(cache=True)
def _authalic_to_geodetic_rad(xi_rad, inv_coeffs):
    return xi_rad + fourier_sin(2.0 * xi_rad, inv_coeffs)

# Precomputed coefficients for WGS84 for performance
WGS84_FWD, WGS84_INV, WGS84_QN = compute_fourier_coeffs(WGS84_F, AUTHALIC_FWD, AUTHALIC_INV)

@nb.njit(cache=True)
def geodetic_to_authalic(phi, degrees=True):
    """
    Convert geodetic latitude (ϕ) to authalic latitude (ξ) for WGS84.
    """
    phi_rad = np.radians(phi) if degrees else phi
    xi_rad = _geodetic_to_authalic_rad(phi_rad, WGS84_FWD)
    return np.degrees(xi_rad) if degrees else xi_rad

@nb.njit(cache=True)
def authalic_to_geodetic(xi, degrees=True):
    """
    Convert authalic latitude (ξ) to geodetic latitude (ϕ) for WGS84.
    """
    xi_rad = np.radians(xi) if degrees else xi
    phi_rad = _authalic_to_geodetic_rad(xi_rad, WGS84_INV)
    return np.degrees(phi_rad) if degrees else phi_rad

# Generic versions that take ellipsoid parameters
@nb.njit(cache=True)
def geodetic_to_authalic_custom(phi, f, poly_fwd, degrees=True):
    phi_rad = np.radians(phi) if degrees else phi
    n = third_flattening(f)
    fwd = np.empty(POLYNOMIAL_ORDER, dtype=np.float64)
    for i in range(POLYNOMIAL_ORDER):
        fwd[i] = n * horner(n, poly_fwd[i])
    xi_rad = _geodetic_to_authalic_rad(phi_rad, fwd)
    return np.degrees(xi_rad) if degrees else xi_rad

@nb.njit(cache=True)
def authalic_to_geodetic_custom(xi, f, poly_inv, degrees=True):
    xi_rad = np.radians(xi) if degrees else xi
    n = third_flattening(f)
    inv = np.empty(POLYNOMIAL_ORDER, dtype=np.float64)
    for i in range(POLYNOMIAL_ORDER):
        inv[i] = n * horner(n, poly_inv[i])
    phi_rad = _authalic_to_geodetic_rad(xi_rad, inv)
    return np.degrees(phi_rad) if degrees else phi_rad
