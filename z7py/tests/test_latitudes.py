import math
import pytest
import numpy as np
from z7py import geodetic_to_authalic, authalic_to_geodetic

def test_poles_and_equator():
    # Equator
    assert math.isclose(geodetic_to_authalic(0.0), 0.0, abs_tol=1e-12)
    assert math.isclose(authalic_to_geodetic(0.0), 0.0, abs_tol=1e-12)
    
    # North Pole
    assert math.isclose(geodetic_to_authalic(90.0), 90.0, abs_tol=1e-12)
    assert math.isclose(authalic_to_geodetic(90.0), 90.0, abs_tol=1e-12)
    
    # South Pole
    assert math.isclose(geodetic_to_authalic(-90.0), -90.0, abs_tol=1e-12)
    assert math.isclose(authalic_to_geodetic(-90.0), -90.0, abs_tol=1e-12)

def test_round_trip():
    latitudes = [-80, -45, -10, 10, 45, 80]
    for lat in latitudes:
        xi = geodetic_to_authalic(lat)
        phi = authalic_to_geodetic(xi)
        assert math.isclose(lat, phi, abs_tol=1e-12)

def test_radians():
    lat_rad = math.radians(45.0)
    xi_rad = geodetic_to_authalic(lat_rad, degrees=False)
    phi_rad = authalic_to_geodetic(xi_rad, degrees=False)
    assert math.isclose(lat_rad, phi_rad, abs_tol=1e-15)

def test_array_input():
    # Numba-jitted functions in latitudes.py take scalars, 
    # but we can wrap them or use np.vectorize if we wanted.
    # Currently they are defined for scalars.
    lats = np.array([0.0, 45.0, 90.0])
    # Test if we can call them in a loop
    xis = np.array([geodetic_to_authalic(lat) for lat in lats])
    assert len(xis) == 3
    assert math.isclose(xis[0], 0.0)
    assert xis[1] < 45.0 # Authalic latitude is slightly less than geodetic (except at poles/equator)
    assert math.isclose(xis[2], 90.0)
