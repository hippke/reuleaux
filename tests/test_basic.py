import math
import reuleaux as rx

C = ((0, 0, 1), (0.8, 0.3, 0.6), (0.5, -0.4, 0.7))


def test_area_matches_reference():
    assert abs(rx.overlap_area(*C) - 0.357820532455879) < 1e-14
    assert abs(rx.overlap_area(*C) - rx.reference_area(*C)[0]) < 1e-12


def test_equal_circles_reuleaux_triangle():
    # three unit circles through each other's centres -> Reuleaux triangle
    s = math.sqrt(3) / 2
    a = rx.overlap_area((0, 0, 1), (1, 0, 1), (0.5, s, 1))
    assert abs(a - 0.5 * (math.pi - math.sqrt(3))) < 1e-14


def test_disjoint_and_nested():
    assert rx.overlap_area((0, 0, 1), (5, 0, 1), (0, 5, 1)) == 0.0
    assert abs(rx.overlap_area((0, 0, 1), (0, 0, 2), (0, 0, 3)) - math.pi) < 1e-14


def test_selftest():
    assert rx.selftest(500, verbose=False)
