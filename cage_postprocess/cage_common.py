'''Shared helpers for comparing CAGE floor-plan output with uLayout estimates.

CAGE (`align_floorplan.py`) stores each room's corners in `world_mm` in the
"ps" projection frame: the input point cloud after up-axis reorder (up moved to
column 2), yaw correction (`applied_yaw_deg`, already applied) and the
`ps = -xyz; ps[0] *= -1; ps[1] *= -1` sign flips (which cancel for the two
floor-plane columns, leaving ps_z = -height). For MVS scenes `world_mm`
actually holds METERS (see CAGE eval_floorplan.md).

This module maps those corners back into the original point-cloud world frame
(= the COLMAP SfM world when the cloud came from the same reconstruction) and
verifies that assumption against the sparse points3D percentiles. Mirrors
CAGE/polys_to_3d.py, which is the authoritative inverse.
'''
import json

import numpy as np

# Forward column permutation of CAGE reorder_up_axis: reordered = orig[:, PERM]
PERM = {'z': [0, 1, 2], 'y': [0, 2, 1], 'x': [1, 2, 0]}


def inverse_perm(up_axis):
    p = PERM[up_axis]
    inv = [0, 0, 0]
    for k, src in enumerate(p):
        inv[src] = k
    return inv


def undo_yaw(xy, yaw_deg):
    '''Rotate 2D floor-plane points by -yaw_deg (inverse of rotate_floor_plane).'''
    r = np.deg2rad(-yaw_deg)
    c, s = np.cos(r), np.sin(r)
    xy = np.asarray(xy, dtype=np.float64)
    return np.stack([xy[:, 0] * c - xy[:, 1] * s,
                     xy[:, 0] * s + xy[:, 1] * c], axis=1)


def ps_to_world(xy, yaw_deg, up_axis, height=0.0):
    '''CAGE ps floor-plane corners [N,2] -> original world 3D [N,3] at height.'''
    r01 = undo_yaw(xy, yaw_deg)
    h = np.full((len(r01), 1), float(height))
    reordered = np.concatenate([r01, h], axis=1)
    return reordered[:, inverse_perm(up_axis)]


def plan_axes(up_axis):
    '''Original-frame column indices of the floor plane, in reorder order.

    For the default y-up: (0, 2) -> the world (X, Z) plane used by
    boundary_to_floorplan / multi_layout_viewer.
    '''
    p = PERM[up_axis]
    return p[0], p[1]


def load_cage(json_path):
    '''Load a CAGE *_aligned_polys.json -> rooms in the original world frame.

    Returns dict:
      rooms_xz   list of [N,2] float64 room polygons, original-world plan coords
      centroids  [R,2] polygon centroids (area-weighted)
      y_floor, y_ceil  floor/ceiling height along the up axis (original world)
      norm, yaw, up_axis, raw (full json)
    `world_mm` is read as meters (MVS convention).
    '''
    data = json.load(open(json_path))
    norm = data['normalization']
    yaw = float(norm.get('applied_yaw_deg', 0.0))
    up_axis = norm.get('up_axis', 'y')
    ax, az = plan_axes(up_axis)

    rooms_xz, centroids = [], []
    for room in data['rooms']:
        ps_xy = np.asarray(room['world_mm'], dtype=np.float64)  # meters (MVS)
        w3 = ps_to_world(ps_xy, yaw, up_axis)
        xz = w3[:, [ax, az]]
        rooms_xz.append(xz)
        centroids.append(_poly_centroid(xz))

    # Heights live in ps column 2 (ps_z = -height). In THIS COLMAP world gravity
    # points along +Y (world up is -Y): the physical FLOOR is the plane with the
    # larger world Y (~0.0) and the CEILING the smaller (~-2.4) -- verified by
    # projecting the gaussian cloud into a real pano (floor lands at image
    # bottom only when -Y is treated as up). So the floor takes the LOW ps
    # percentile and the ceiling the HIGH one. Consumers compute the camera
    # height above the floor as (y_floor - cam_y), positive because up is -Y.
    pct_lo = np.asarray(norm['coords_pct_low'], dtype=np.float64)
    pct_hi = np.asarray(norm['coords_pct_high'], dtype=np.float64)
    y_floor = -float(pct_lo[2])          # larger world Y  -> physical floor
    y_ceil = -float(pct_hi[2])           # smaller world Y -> physical ceiling

    return {'rooms_xz': rooms_xz, 'centroids': np.asarray(centroids),
            'y_floor': y_floor, 'y_ceil': y_ceil,
            'norm': norm, 'yaw': yaw, 'up_axis': up_axis, 'raw': data}


def _poly_centroid(xz):
    x, z = xz[:, 0], xz[:, 1]
    xn, zn = np.roll(x, -1), np.roll(z, -1)
    cross = x * zn - xn * z
    a = cross.sum() / 2.0
    if abs(a) < 1e-9:
        return xz.mean(axis=0)
    cx = ((x + xn) * cross).sum() / (6.0 * a)
    cz = ((z + zn) * cross).sum() / (6.0 * a)
    return np.array([cx, cz])


def check_same_world(xyz, norm, shift_tol=0.5, scale_tol=0.03):
    '''Verify the sparse COLMAP cloud and the CAGE input cloud share one world.

    CAGE records `coords_pct_low/high` of its input cloud with columns 0/1
    PRE-yaw in the reordered frame and column 2 sign-flipped (ps_z = -height)
    (infer_pointcloud.py:180-183). Reproduce that from the sparse points and
    compare per axis. Raw percentile residuals are only informative -- sparse
    SfM and dense MVS clouds have different tail densities, so a few tens of
    cm there is normal. The hard gate uses the robust invariants: the per-axis
    SPAN ratio (scale) and the interval MIDPOINT shift (translation).
    '''
    up_axis = norm.get('up_axis', 'y')
    pct = norm.get('percentile') or {'low': 2.0, 'high': 98.0}
    reord = np.asarray(xyz, dtype=np.float64)[:, PERM[up_axis]]
    lo = np.percentile(reord, pct['low'], axis=0)
    hi = np.percentile(reord, pct['high'], axis=0)
    got_lo = np.array([lo[0], lo[1], -hi[2]])
    got_hi = np.array([hi[0], hi[1], -lo[2]])
    ref_lo = np.asarray(norm['coords_pct_low'], dtype=np.float64)
    ref_hi = np.asarray(norm['coords_pct_high'], dtype=np.float64)
    res_lo = got_lo - ref_lo
    res_hi = got_hi - ref_hi
    print('[cage] same-world check (sparse pct%.0f/%.0f vs CAGE json):'
          % (pct['low'], pct['high']))
    for name, r_ref, r_got, r in (('low', ref_lo, got_lo, res_lo),
                                  ('high', ref_hi, got_hi, res_hi)):
        print('  %-4s json (%8.3f %8.3f %8.3f)  sparse (%8.3f %8.3f %8.3f)  '
              'resid (%+.3f %+.3f %+.3f)'
              % ((name,) + tuple(r_ref) + tuple(r_got) + tuple(r)))
    ratio = (got_hi - got_lo) / (ref_hi - ref_lo)
    shift = ((got_lo + got_hi) - (ref_lo + ref_hi)) / 2.0
    print('  span ratio (%.3f %.3f %.3f)  midpoint shift (%+.3f %+.3f %+.3f) m'
          % (tuple(ratio) + tuple(shift)))
    # gate on the two floor-plane axes only: sparse SfM features are scarce at
    # the exact floor/ceiling planes, so the height percentiles sit inward of
    # the dense-MVS ones (span visibly < 1) without implying a frame mismatch
    bad_scale = float(np.abs(ratio[:2] - 1.0).max())
    bad_shift = float(np.abs(shift[:2]).max())
    if bad_scale > scale_tol or bad_shift > shift_tol:
        raise RuntimeError(
            'sparse/0 points and the CAGE input cloud disagree (scale off by '
            '%.1f%%, shift %.2f m): not the same world -- registration needed'
            % (bad_scale * 100, bad_shift))
    print('  -> same world/scale (scale within %.1f%%, shift %.2f m)'
          % (bad_scale * 100, bad_shift))
    return True
