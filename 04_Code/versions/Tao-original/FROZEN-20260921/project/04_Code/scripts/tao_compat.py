# -*- coding: utf-8 -*-
"""
Compatibility shims for the released Tao package under numpy >= 2
(strict in-place casting). NO algorithm change: the original functions mutate
int16 contour points with float steps, which old numpy silently upcast; we
pass float copies instead. Callers keep their int arrays for indexing.
"""
import os
import sys
from pathlib import Path
import numpy as np

TAO = os.environ.get('GREAC_TAO_CODE', str(Path(__file__).resolve().parents[2] / '10_References/Tao_source_code/scripts'))
if TAO not in sys.path:
    sys.path.insert(0, TAO)
from lake_bathymetry.geometry_utils import (EightNeighbor,               # noqa
                                            draw_extended_line as _del,
                                            TotalDistance as _td)


def draw_extended_line(binaryimage, v1, v2):
    return _del(binaryimage, np.asarray(v1, dtype='float64'),
                np.asarray(v2, dtype='float64'))


def TotalDistance(slope, binaryimage, point_inside, vector):
    return _td(slope, binaryimage,
               np.asarray(point_inside, dtype='float64'),
               np.asarray(vector, dtype='float64'))


def TotalDistance_wmax(slope, binaryimage, point_inside, vector, wmax_px):
    """W_max-capped variant of TotalDistance (width-correction ablation).

    Physics: the shore slope informs the underwater profile only within a
    finite correlation length W_max. We still WALK to the real banks to fetch
    their measured slopes, but the profile GEOMETRY (endpoints, width, lowest
    point) is capped at wmax_px steps per side. So on a 20-km-wide basin the
    parabola is built over <= 2*W_max instead of the full width, and the bank
    slope is no longer diluted by the lake's total width.
    Implementation note: original calculate() reads slope[] AT the returned
    endpoints, so we copy the real-bank slope values onto the capped
    endpoints (contained write; those cells' slope is not used elsewhere)."""
    pi = np.asarray(point_inside, dtype='float64')
    v = np.asarray(vector, dtype='float64')
    t1 = pi.copy(); n1 = 0
    while binaryimage[int(t1[0]), int(t1[1])] != 0:
        t1 += v; n1 += 1
    t2 = pi.copy(); n2 = 0
    while binaryimage[int(t2[0]), int(t2[1])] != 0:
        t2 -= v; n2 += 1
    p1_bank = np.int16(t1 + v)     # real-bank slope anchors
    p2_bank = np.int16(t2 - v)
    s1 = slope[p1_bank[0], p1_bank[1]]
    s2 = slope[p2_bank[0], p2_bank[1]]
    n1c, n2c = min(n1, wmax_px), min(n2, wmax_px)
    p1 = np.int16(pi + v * (n1c + 1))   # capped geometric endpoints
    p2 = np.int16(pi - v * (n2c + 1))
    if slope[p1[0], p1[1]] == 0:
        slope[p1[0], p1[1]] = s1
    if slope[p2[0], p2[1]] == 0:
        slope[p2[0], p2[1]] = s2
    length = n1c + n2c
    if s1 == 0 and s2 == 0:
        weight = 0.5
    else:
        if s1 > 3 * s2:
            s2 = s1
        elif s2 > 3 * s1:
            s1 = s2
        weight = s1 / (s1 + s2)
    current_len = max(1, int(weight * length))
    point_lowest = np.int16(pi - v * n2c + v * current_len)
    return length, p1, p2, point_lowest
