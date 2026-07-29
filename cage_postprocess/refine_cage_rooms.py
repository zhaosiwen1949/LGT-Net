'''Refine the CAGE room segmentation using LGT-Net per-pano BEV evidence.

Fixes two CAGE failure modes, driven by the per-frame layout rings that
infer_gsplat_panos.py + cage_bev_compare.frame_ring produce:

  under-segmentation  one CAGE room actually holds several real rooms.
      Detected by grouping the room's frame rings into connected components
      of "rings overlap" (group_frames): >= 2 mutually exclusive components
      that are separable along one Manhattan axis mean the room is split at
      the seam between them (snapped to existing CAGE wall lines when within
      --snap-tol).
  over-segmentation   one real room was cut into several CAGE rooms.
      Candidates come from CAGE's own metadata (align_info.splits records,
      micro rooms nobody ever walked into); the decision needs the LGT
      rings of a neighbour to spill over most of the candidate room.

Both directions are grounded on the REAL SfM camera trajectory
(sparse/0/frames.bin, 2661 poses here vs 75 virtual viewpoints): a region a
person physically walked through is a room, a region nobody entered is not.
That physical fact replaces the arbitrary "a sub-room must be >= X m2" floor
on the split side, and settles the merge direction on the other.

GT is never consulted; CAGE metadata only generates/corroborates candidates,
the LGT rings decide. Conservative gates keep normal rooms (rings that are
merely 10-25% oversized, doorway spill like r8->r0) untouched.

Grouping happens at FRAME level rather than viewpoint-cluster level, which is
what lets a sub-room seen by a single frame be detected at all, and which also
makes doorway cameras a non-issue -- see group_frames().

All decisions and geometry run in the CAGE "ps" frame (axis-aligned room
polygons; Manhattan directions == coordinate axes); the authoritative output
geometry is the integer `pixel` polygon (what CAGE eval_floorplan.py reads),
with world_mm/world_m regenerated from it exactly like the original file.

Outputs (none overwrite the originals):
  <floorplan>/<scene>_..._refined_polys.json   corrected CAGE json
  <out_dir>/refine_report.json                 evidence + decisions
  <out_dir>/refined_selection.json             frames re-assigned to the
                                               refined rooms (for
                                               cage_bev_compare --selection)
  <out_dir>/refine_overlay.png                 before/after comparison

Run inside the LGT-Net conda env from the repo root:
    python cage_postprocess/refine_cage_rooms.py \
        --infer_dir src/output/huizhongbeili-106/gsplat_infer_zind_traj \
        --dry_run
'''
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, _ROOT)

import cage_common as cc
import traj_sampling as ts
from cage_bev_compare import frame_ring, to_poly

FRAME_RE = re.compile(r'^r(\d+)_c(\d+)_s(\d+)$')


# --------------------------------------------------------------------------- #
# pixel <-> ps transforms (mirror CAGE eval_floorplan.pixel_to_metres)
# --------------------------------------------------------------------------- #

def pixel_to_ps(cols, rows, norm):
    '''Integer pixel (col,row) -> ps-frame meters. unit_scale=1 (MVS).'''
    mn = np.asarray(norm['min_coords'], dtype=np.float64)
    mx = np.asarray(norm['max_coords'], dtype=np.float64)
    cols = np.asarray(cols, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.float64)
    if bool(norm.get('hflip', False)):
        cols = 255. - cols
    wx = mn[0] + (cols / 255.) * (mx[0] - mn[0])
    wy = mn[1] + (rows / 255.) * (mx[1] - mn[1])
    return np.stack([wx, wy], axis=1)


def ps_to_pixel(xy, norm):
    '''ps-frame meters [N,2] -> float pixel (col,row) [N,2].'''
    mn = np.asarray(norm['min_coords'], dtype=np.float64)
    mx = np.asarray(norm['max_coords'], dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float64)
    cols = (xy[:, 0] - mn[0]) / (mx[0] - mn[0]) * 255.
    if bool(norm.get('hflip', False)):
        cols = 255. - cols
    rows = (xy[:, 1] - mn[1]) / (mx[1] - mn[1]) * 255.
    return np.stack([cols, rows], axis=1)


def ps_step(norm, axis):
    '''ps meters spanned by one pixel along ps axis 0 (col) / 1 (row).'''
    mn, mx = norm['min_coords'], norm['max_coords']
    return (mx[axis] - mn[axis]) / 255.


# --------------------------------------------------------------------------- #
# polygon helpers (pixel frame, integer axis-aligned rings)
# --------------------------------------------------------------------------- #

def px_poly(pixel):
    from shapely.geometry import Polygon
    return Polygon(np.asarray(pixel, dtype=np.float64))


def ring_from_poly(poly):
    '''Shapely polygon (integer-valued coords) -> clockwise int ring, no
    closing duplicate, collinear vertices removed (matches file convention:
    negative shoelace in (col,row)).'''
    xy = np.asarray(poly.exterior.coords[:-1], dtype=np.float64)
    r = np.round(xy)
    assert np.abs(xy - r).max() < 1e-6, 'non-integer vertex after cut: %s' % xy
    xy = r.astype(int)
    # drop collinear vertices
    keep = []
    n = len(xy)
    for i in range(n):
        a, b, c = xy[(i - 1) % n], xy[i], xy[(i + 1) % n]
        if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) != 0:
            keep.append(i)
    xy = xy[keep]
    x, y = xy[:, 0], xy[:, 1]
    signed = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if signed > 0:
        xy = xy[::-1]
    return [[int(c), int(r)] for c, r in xy]


def halfplane_clip(poly, axis, c, side, pad=1e3):
    '''poly ∩ {p: p[axis] <= c} (side=-1) or >= c (side=+1).'''
    from shapely.geometry import box
    x0, y0, x1, y1 = poly.bounds
    x0 -= pad; y0 -= pad; x1 += pad; y1 += pad
    if axis == 0:
        b = box(x0, y0, c, y1) if side < 0 else box(c, y0, x1, y1)
    else:
        b = box(x0, y0, x1, c) if side < 0 else box(x0, c, x1, y1)
    return poly.intersection(b)


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #

def group_frames(rings, t_link):
    '''{frame name: ring} -> ([{names, poly}], min_cross_ov), largest first.

    Connected components over "rings overlap by >= t_link" (overlap normalized
    by the smaller ring, so a small closet ring still links to the big room
    ring it sits inside). Frame level, NOT viewpoint-cluster level: a sub-room
    that only ONE frame ever saw still forms its own component, whereas
    clustering by camera position buries it inside whichever cluster the
    sampler happened to put it in.

    This also removes the need to detect doorway cameras separately. A frame
    shot from an opening predicts a layout dominated by one of the two spaces,
    so it simply joins that side's component (r00_c2_s1 overlaps all six
    bedroom frames by 0.66-0.93 and both corridor frames by 0.00); a frame
    standing in a genuinely different room overlaps nothing and forms its own
    (r01_c0_s1, degree 0). Both land in the right place with no special case.
    '''
    from shapely.ops import unary_union
    names = sorted(rings)
    parent = {n: n for n in names}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    ov = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb = rings[a], rings[b]
            o = pa.intersection(pb).area / max(min(pa.area, pb.area), 1e-9)
            ov[(a, b)] = o
            if o >= t_link:
                parent[find(a)] = find(b)
    groups = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    out = [{'names': sorted(g), 'poly': unary_union([rings[n] for n in g])}
           for g in groups.values()]
    out.sort(key=lambda g: -g['poly'].area)
    return out, ov


def count_traj(poly, traj_pts):
    '''Number of real trajectory positions (ps frame Points) inside poly.'''
    return sum(1 for p in traj_pts if poly.contains(p))


def detect_split(rid, poly_ps, rings, cams, traj_pts, args):
    '''One room -> (op dict or None, evidence dict).'''
    groups, ov = group_frames(rings, args.t_link)
    ev = {'room': rid, 'n_frames': len(rings),
          'groups': [[n.split('_', 1)[1] for n in g['names']]
                     for g in groups]}
    if len(groups) > 1:
        a, b = set(groups[0]['names']), set(groups[1]['names'])
        pairs = [v for (x, y), v in ov.items()
                 if {x, y} & a and {x, y} & b]
        ev['max_frame_ov_A_B'] = round(max(pairs, default=0.0), 3)
    if len(groups) < 2:
        ev['verdict'] = 'single group'
        return None, ev
    A, B = groups[0], groups[1]
    if len(groups) > 2:
        ev['note'] = 'more than 2 groups; only the top-2 considered'
    cross = A['poly'].intersection(B['poly']).area / \
        max(min(A['poly'].area, B['poly'].area), 1e-9)
    ev['g1_cross_ov'] = round(cross, 3)
    if cross >= args.t_cross:
        ev['verdict'] = 'ambiguous (cross overlap %.2f in [%.2f, %.2f))' % (
            cross, args.t_cross, args.t_link)
        return None, ev
    in_a = A['poly'].intersection(poly_ps).area / max(A['poly'].area, 1e-9)
    in_b = B['poly'].intersection(poly_ps).area / max(B['poly'].area, 1e-9)
    ev['g2_inside'] = [round(in_a, 3), round(in_b, 3)]
    if min(in_a, in_b) < args.g2_inside:
        ev['verdict'] = 'group mostly outside the parent room'
        return None, ev
    S_A = A['poly'].intersection(poly_ps)
    S_B = B['poly'].intersection(poly_ps)
    E_A = S_A.difference(B['poly'])
    E_B = S_B.difference(A['poly'])
    # G3: each exclusive region was physically walked through. This replaces
    # the old "exclusive area >= 0.8 m2" floor, which was an arbitrary number
    # that a real 1 m2 walk-in closet only barely cleared. The real SfM
    # trajectory is both denser and more meaningful: every genuine room here
    # carries 13-28 poses/m2, a prediction sliver carries none.
    ev['g3_exclusive_m2'] = [round(E_A.area, 2), round(E_B.area, 2)]
    traj_in = [count_traj(E_A, traj_pts), count_traj(E_B, traj_pts)]
    ev['g3_traj_in_exclusive'] = traj_in
    if min(traj_in) < args.g3_traj:
        ev['verdict'] = ('exclusive region without real trajectory '
                         '(%d/%d < %d)' % (traj_in[0], traj_in[1],
                                           args.g3_traj))
        return None, ev
    # G3b: the camera that predicted a group stood in that group's own
    # exclusive region -- ties the region to the frames claiming it, which the
    # unlabelled trajectory above cannot do.
    cam_in = []
    for g, E in ((A, E_A), (B, E_B)):
        pad = E.buffer(args.cam_tol)
        cam_in.append(sum(1 for n in g['names']
                          if n in cams and pad.contains(cams[n])))
    ev['g3b_cams_in_exclusive'] = cam_in
    if min(cam_in) < 1:
        ev['verdict'] = 'no camera inside one of the exclusive regions'
        return None, ev
    # G4: separable along one ps axis
    axis, best_ovl = None, None
    for k in (0, 1):
        a0, a1 = E_A.bounds[k], E_A.bounds[k + 2]
        b0, b1 = E_B.bounds[k], E_B.bounds[k + 2]
        ovl = max(0.0, min(a1, b1) - max(a0, b0))
        rel = ovl / max(min(a1 - a0, b1 - b0), 1e-9)
        if best_ovl is None or rel < best_ovl:
            axis, best_ovl = k, rel
    ev['g4_axis'] = 'xy'[axis]
    ev['g4_proj_overlap'] = round(best_ovl, 3)
    if best_ovl > args.g4_ovl:
        ev['verdict'] = 'groups not separable along one axis'
        return None, ev
    ev['verdict'] = 'SPLIT'
    return {'op': 'split', 'room': rid, 'axis': axis,
            'S_A': S_A, 'S_B': S_B,
            'groups': [A['names'], B['names']]}, ev


def detect_merges(raw, polys_ps, room_unions, traj_n, args, norm):
    '''-> (list of merge ops {survivor, absorbed}, evidence list).

    Candidate pairs need a STRUCTURAL prior (a recorded programmatic split,
    or a micro room nobody walked into); the LGT ring spill of a neighbour
    then decides. Ring spill alone never triggers a merge.

    traj_n {room id: real SfM poses inside it} carries the veto that matters:
    a room somebody walked around in is a room, whatever its neighbours'
    rings say. It also fixes the direction of a recorded split, where both
    halves are candidates for absorbing each other.'''
    from shapely.ops import unary_union
    n = len(polys_ps)

    def shared_seg(i, j):
        return polys_ps[i].boundary.intersection(polys_ps[j].boundary)

    def edges_ps(i):
        b = polys_ps[i].bounds
        return min(b[2] - b[0], b[3] - b[1])

    cands, ev = set(), []
    for s in raw.get('align_info', {}).get('splits', []):
        # the two halves of a recorded split share an edge on the split line;
        # add both directions -- the self-coverage veto sorts out which half
        # is the real room
        axis = 1 if s['axis'] == 'y' else 0
        line_ps = pixel_to_ps([s['line'], s['line']],
                              [s['line'], s['line']], norm)[0][axis]
        tol = 3.0 * abs(ps_step(norm, axis))
        for i in range(n):
            for j in range(i + 1, n):
                seg = shared_seg(i, j)
                if seg.is_empty or seg.length < 0.3:
                    continue
                b = seg.bounds
                mid = 0.5 * (b[axis] + b[axis + 2])
                if abs(b[axis + 2] - b[axis]) < 1e-6 \
                        and abs(mid - line_ps) <= tol:
                    cands.add((i, j))
                    cands.add((j, i))
    for t in range(n):
        micro = polys_ps[t].area < args.micro_area or edges_ps(t) < args.micro_edge
        if micro and traj_n.get(t, 0) < args.merge_traj:
            for j in range(n):
                if j != t and shared_seg(t, j).length >= args.share_edge:
                    cands.add((t, j))
    # candidate (t, n): absorb room t into neighbour n, decided by n's rings
    by_target = {}
    for t, nb in sorted(cands):
        by_target.setdefault(t, []).append(nb)
    ops = []
    for t, nbs in sorted(by_target.items()):
        rec = {'target': t, 'neighbours': nbs,
               'traj_in_target': traj_n.get(t, 0)}
        if traj_n.get(t, 0) >= args.merge_traj:
            rec['verdict'] = ('real trajectory walks through the target '
                              '(%d poses) -> real room' % traj_n[t])
            ev.append(rec)
            continue
        own = room_unions.get(t)
        if own is not None:
            self_cover = own.intersection(polys_ps[t]).area / polys_ps[t].area
            rec['self_cover'] = round(self_cover, 3)
            if self_cover >= args.self_cover:
                rec['verdict'] = 'target has own coverage -> real room'
                ev.append(rec)
                continue
        spills = []
        for nb in nbs:
            u = room_unions.get(nb)
            if u is None:
                continue
            spills.append((u.intersection(polys_ps[t]).area /
                           max(polys_ps[t].area, 1e-9), nb))
        spills.sort(reverse=True)
        rec['spill'] = [{'from': nb, 'frac': round(s, 3)} for s, nb in spills]
        if not spills or spills[0][0] < args.spill_thr:
            rec['verdict'] = 'no neighbour spills enough'
            ev.append(rec)
            continue
        if len(spills) > 1 and spills[1][0] >= 0.5 * spills[0][0] \
                and spills[1][0] >= args.spill_thr:
            rec['verdict'] = 'ambiguous: two neighbours spill comparably'
            ev.append(rec)
            continue
        rec['verdict'] = 'MERGE into %d' % spills[0][1]
        ev.append(rec)
        ops.append({'op': 'merge', 'survivor': spills[0][1], 'absorbed': t})
    return ops, ev


# --------------------------------------------------------------------------- #
# split geometry
# --------------------------------------------------------------------------- #

def seam_scan(poly_ps, axis, S_A, S_B, norm, wall_lines, args):
    '''Find the cut coordinate along ps axis; returns (c_ps, diag).'''
    step = abs(ps_step(norm, axis))
    lo = poly_ps.bounds[axis] + step
    hi = poly_ps.bounds[axis + 2] - step
    # which side of the axis does A live on?
    a_c = S_A.centroid.coords[0][axis]
    b_c = S_B.centroid.coords[0][axis]
    a_side = -1 if a_c < b_c else 1
    cs = np.arange(lo, hi + 1e-9, step)
    costs = np.array([
        halfplane_clip(S_A, axis, c, -a_side).area +
        halfplane_clip(S_B, axis, c, a_side).area for c in cs])
    best = costs.min()
    flat = cs[costs <= best + 1e-9]
    c = float(0.5 * (flat.min() + flat.max()))
    diag = {'axis': 'xy'[axis], 'a_side': a_side,
            'scan_lo': round(float(lo), 3), 'scan_hi': round(float(hi), 3),
            'cost_min': round(float(best), 4),
            'plateau': [round(float(flat.min()), 3),
                        round(float(flat.max()), 3)],
            'c_seam': round(c, 3)}
    near = [w for w in wall_lines if abs(w - c) < args.snap_tol]
    if near:
        w = min(near, key=lambda w: abs(w - c))
        diag['snapped_to_wall'] = round(float(w), 3)
        c = float(w)
    diag['c_final_ps'] = round(c, 4)
    return c, a_side, diag


def cut_room(pixel, axis, c_ps, norm):
    '''Cut a room's integer pixel ring at ps coordinate c_ps along ps axis.
    Returns (piece_lo, piece_hi, c_px): shapely pixel polygons on the ps-low /
    ps-high side of the cut, and the integer pixel line.'''
    cpx_f = ps_to_pixel([[c_ps, c_ps]], norm)[0]
    hflip = bool(norm.get('hflip', False))
    if axis == 0:
        c_px = int(round(cpx_f[0]))
        px_axis, flip = 0, hflip
    else:
        c_px = int(round(cpx_f[1]))
        px_axis, flip = 1, False
    poly = px_poly(pixel)
    lo_px = halfplane_clip(poly, px_axis, c_px, -1)
    hi_px = halfplane_clip(poly, px_axis, c_px, +1)
    if flip:                       # larger col == smaller ps x
        lo_px, hi_px = hi_px, lo_px
    assert lo_px.geom_type == 'Polygon' and hi_px.geom_type == 'Polygon', \
        'cut produced %s / %s' % (lo_px.geom_type, hi_px.geom_type)
    assert abs(lo_px.area + hi_px.area - poly.area) < 1e-6, 'area not conserved'
    return lo_px, hi_px, c_px


# --------------------------------------------------------------------------- #
# openings remap
# --------------------------------------------------------------------------- #

def opening_seg_px(op):
    from shapely.geometry import LineString
    s, e = op['span']
    if op['axis'] == 'x':          # fixed column, span along rows
        return LineString([(op['line'], s), (op['line'], e)])
    return LineString([(s, op['line']), (e, op['line'])])


def remap_openings(openings, merge_map, split_pieces, warns):
    '''merge_map: {absorbed: survivor}; split_pieces: {old_id: [(new_id,
    pixel_poly), ...]}. Returns the remapped list (deep-copied).'''
    out = []
    for op in openings:
        op = json.loads(json.dumps(op))
        rooms = [merge_map.get(r, r) for r in op['rooms']]
        rooms = list(dict.fromkeys(rooms))
        for old_id, pieces in split_pieces.items():
            if old_id not in op['rooms']:
                continue
            seg = opening_seg_px(op)
            best = []
            for new_id, piece in pieces:
                ovl = seg.intersection(piece.boundary.buffer(1.5)).area \
                    if seg.length == 0 else \
                    seg.intersection(piece.boundary.buffer(1.5)).length
                best.append((ovl, new_id))
            best.sort(reverse=True)
            if len(best) > 1 and best[1][0] > 0.4 * max(best[0][0], 1e-9) \
                    and best[1][0] > 1.0:
                warns.append('opening %s straddles the new wall; assigned '
                             'to room %d' % (op, best[0][1]))
            rooms = [best[0][1] if r == old_id else r for r in rooms]
        op['rooms'] = rooms
        out.append(op)
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dataset_dir', default='src/datasets/huizhongbeili-106')
    ap.add_argument('--infer_dir',
                    default='src/output/huizhongbeili-106/gsplat_infer_zind_traj')
    ap.add_argument('--cage_json', default=None)
    ap.add_argument('--render_dir', default=None)
    ap.add_argument('--sparse_dir', default=None,
                    help='COLMAP sparse model holding the real trajectory '
                         '(default <dataset_dir>/sparse/0)')
    ap.add_argument('--h-band', type=float, nargs=2, default=(0.9, 1.9),
                    help='keep trajectory poses this far above the gaussian '
                         'floor (drops bogus/handheld-low poses)')
    ap.add_argument('--out_json', default=None,
                    help='refined CAGE json (default: cage json with '
                         '_aligned_polys.json -> _refined_polys.json)')
    ap.add_argument('--out_dir', default=None,
                    help='report/selection/overlay dir (default src/output/'
                         '<scene>/refine_<infer suffix>)')
    ap.add_argument('--dry_run', action='store_true',
                    help='detection + evidence report only, write nothing')
    ap.add_argument('--max-radius', type=float, default=12.0)
    # split gates
    ap.add_argument('--t-link', type=float, default=0.30)
    ap.add_argument('--t-cross', type=float, default=0.15)
    ap.add_argument('--g2-inside', type=float, default=0.40)
    ap.add_argument('--g3-traj', type=int, default=5,
                    help='real SfM poses required inside each exclusive '
                         'region; 5 poses ~= 0.85 m of walking at the 0.17 m '
                         'median frame spacing. Replaces the old area floor: '
                         'size does not make a room, being walked into does')
    ap.add_argument('--g4-ovl', type=float, default=0.20)
    ap.add_argument('--g5-side', type=float, default=0.75)
    ap.add_argument('--cam-tol', type=float, default=0.25,
                    help='slack (m) when testing whether a group camera lies '
                         'in its own exclusive region')
    ap.add_argument('--snap-tol', type=float, default=0.40)
    # merge gates
    ap.add_argument('--micro-area', type=float, default=1.5)
    ap.add_argument('--micro-edge', type=float, default=0.7)
    ap.add_argument('--share-edge', type=float, default=0.6)
    ap.add_argument('--spill-thr', type=float, default=0.5)
    ap.add_argument('--self-cover', type=float, default=0.5)
    ap.add_argument('--merge-traj', type=int, default=5,
                    help='a room with at least this many real SfM poses is '
                         'never absorbed into a neighbour, and never becomes '
                         'a micro-room candidate')
    return ap.parse_args()


def main():
    from shapely.geometry import Point
    from shapely.ops import unary_union

    args = parse_args()
    ddir = args.dataset_dir
    scene = os.path.basename(os.path.normpath(ddir))
    index = json.load(open(os.path.join(args.infer_dir, 'index.json')))
    render_dir = (args.render_dir or index.get('render_dir')
                  or os.path.join(ddir, 'gsplat_render'))
    sel = json.load(open(os.path.join(render_dir, 'selection.json')))
    cage_json = args.cage_json or os.path.join(
        ddir, 'floorplan', os.path.basename(sel['cage_json']))
    cage = cc.load_cage(cage_json)
    raw = cage['raw']
    norm = raw['normalization']
    yaw = cage['yaw']
    pdata = json.load(open(os.path.join(render_dir,
                                        'virtual_camera_poses.json')))
    poses, meta = pdata['poses'], pdata['metadata']
    floor_y = float(meta['floor_y'])

    out_json = args.out_json or cage_json.replace('_aligned_polys.json',
                                                  '_refined_polys.json')
    assert out_json != cage_json, 'refined name must differ from the source'
    suffix = os.path.basename(args.infer_dir.rstrip('/')) \
        .replace('gsplat_infer_', '')
    out_dir = args.out_dir or os.path.join('src', 'output', scene,
                                           'refine_%s' % suffix)

    # ---- transforms self-check on every stored room -----------------------
    for i, room in enumerate(raw['rooms']):
        px = np.asarray(room['pixel'], dtype=np.float64)
        ps = pixel_to_ps(px[:, 0], px[:, 1], norm)
        err = np.abs(ps - np.asarray(room['world_mm'])).max()
        assert err < 1e-9, 'room %d pixel->ps mismatch %.2e' % (i, err)
        back = ps_to_pixel(ps, norm)
        assert np.abs(back - px).max() < 1e-6, 'room %d ps->pixel roundtrip' % i
    xz0 = cage['rooms_xz'][0]
    assert np.abs(cc.world_to_ps(xz0, yaw)
                  - np.asarray(raw['rooms'][0]['world_mm'])).max() < 1e-9
    print('[refine] transform self-checks passed (%d rooms)' % len(raw['rooms']))

    polys_ps = [to_poly(r['world_mm']) for r in raw['rooms']]

    # ---- per-frame rings in the ps frame ----------------------------------
    rings_ps = {}                      # frame name -> shapely poly (ps)
    frame_xz = {}                      # frame name -> world (x, z)
    room_rings = {}                    # room id -> {frame name: ring}
    for room in sel['rooms']:
        rid = room['id']
        for fr in room['frames']:
            name = fr['frame']
            xz, diag = frame_ring(name, args.infer_dir, poses, floor_y,
                                  max_radius=args.max_radius)
            if xz is None:
                print('[refine] warn: missing ring for %s' % name)
                continue
            p = to_poly(cc.world_to_ps(xz, yaw))
            rings_ps[name] = p
            frame_xz[name] = fr['xz']
            room_rings.setdefault(rid, {})[name] = p
    print('[refine] %d frames -> rings in ps frame' % len(rings_ps))
    room_unions = {rid: unary_union(list(r.values()))
                   for rid, r in room_rings.items()}
    cams_ps = {n: Point(*cc.world_to_ps(np.asarray([xz]), yaw)[0])
               for n, xz in frame_xz.items()}

    # ---- real SfM trajectory (the physical "was anybody here?" evidence) ---
    sparse_dir = args.sparse_dir or os.path.join(ddir, 'sparse', '0')
    _tnames, tC = ts.load_trajectory(sparse_dir)
    th = floor_y - tC[:, 1]                     # world up is -Y
    keep = (th >= args.h_band[0]) & (th <= args.h_band[1])
    traj_pts = [Point(*p) for p in cc.world_to_ps(tC[keep][:, [0, 2]], yaw)]
    traj_n = {rid: count_traj(p, traj_pts) for rid, p in enumerate(polys_ps)}
    print('[refine] %d/%d trajectory poses in the %.1f-%.1f m height band; '
          'per-room counts %s'
          % (len(traj_pts), len(tC), args.h_band[0], args.h_band[1],
             json.dumps({'r%d' % k: v for k, v in traj_n.items()})))

    # ---- detection --------------------------------------------------------
    split_ops, split_ev = [], []
    for rid in sorted(room_rings):
        rings = room_rings[rid]
        if len(rings) < 2:
            split_ev.append({'room': rid, 'n_frames': len(rings),
                             'verdict': 'fewer than 2 frames -> insufficient '
                                        'LGT evidence, skipped'})
            continue
        op, ev = detect_split(rid, polys_ps[rid], rings,
                              {n: cams_ps[n] for n in rings}, traj_pts, args)
        split_ev.append(ev)
        if op:
            split_ops.append(op)
    merge_ops, merge_ev = detect_merges(raw, polys_ps, room_unions, traj_n,
                                        args, norm)
    # frame -> "<room>g<n>" ring group, recorded in the refined selection so a
    # reader can see which frames voted together
    frame_group = {}
    for ev in split_ev:
        for gi, shorts in enumerate(ev.get('groups', [])):
            for s in shorts:
                frame_group['r%02d_%s' % (ev['room'], s)] = 'r%dg%d' % (
                    ev['room'], gi)

    print('\n[refine] split evidence:')
    for ev in split_ev:
        print('  r%-2d %s' % (ev['room'], json.dumps(
            {k: v for k, v in ev.items() if k != 'room'}, ensure_ascii=False)))
    print('[refine] merge evidence:')
    for ev in merge_ev:
        print('  r%-2d %s' % (ev['target'], json.dumps(
            {k: v for k, v in ev.items() if k != 'target'},
            ensure_ascii=False)))
    print('[refine] ops: %s'
          % (['split r%d' % o['room'] for o in split_ops]
             + ['merge r%d->r%d' % (o['absorbed'], o['survivor'])
                for o in merge_ops]))
    if args.dry_run:
        print('[refine] dry run, nothing written')
        return

    # ---- apply: merges first (free id slots), then splits -----------------
    warns = []
    rooms_out = [dict(r) for r in raw['rooms']]     # index == room id
    changed = set()
    merge_map = {}
    for op in merge_ops:
        s, t = op['survivor'], op['absorbed']
        u = px_poly(rooms_out[s]['pixel']).union(px_poly(rooms_out[t]['pixel']))
        assert u.geom_type == 'Polygon' and not list(u.interiors), \
            'merge r%d+r%d not a simple polygon' % (s, t)
        rooms_out[s] = {'pixel': ring_from_poly(u)}
        rooms_out[t] = None
        changed.add(s)
        merge_map[t] = s
    freed = sorted(t for t in merge_map)

    split_pieces = {}                   # old id -> [(new id, pixel poly)]
    id_map_ops = {}
    for op in merge_ops:
        id_map_ops[op['absorbed']] = {'op': 'merge', 'into': op['survivor']}
        id_map_ops.setdefault(op['survivor'],
                              {'op': 'merge', 'from': [op['survivor'],
                                                       op['absorbed']]})
    frame_room_map = dict(merge_map)    # old room id -> new id (simple cases)

    for op in split_ops:
        rid, axis = op['room'], op['axis']
        wall_lines = sorted({round(float(v), 4)
                             for p in polys_ps for v in
                             np.asarray(p.exterior.coords)[:, axis]})
        c_ps, a_side, diag = seam_scan(polys_ps[rid], axis, op['S_A'],
                                       op['S_B'], norm, wall_lines, args)
        lo_px, hi_px, c_px = cut_room(rooms_out[rid]['pixel'], axis, c_ps,
                                      norm)
        # pieces back to ps to decide which keeps the original id (majority
        # of S_A) and to run the G5 consistency gate
        def to_ps_poly(p):
            v = np.asarray(p.exterior.coords[:-1])
            return to_poly(pixel_to_ps(v[:, 0], v[:, 1], norm))
        lo_ps, hi_ps = to_ps_poly(lo_px), to_ps_poly(hi_px)
        a_piece_px, b_piece_px = (lo_px, hi_px) if a_side < 0 else (hi_px, lo_px)
        a_piece_ps, b_piece_ps = (lo_ps, hi_ps) if a_side < 0 else (hi_ps, lo_ps)
        g5_a = op['S_A'].intersection(a_piece_ps).area / op['S_A'].area
        g5_b = op['S_B'].intersection(b_piece_ps).area / op['S_B'].area
        diag['g5_side'] = [round(g5_a, 3), round(g5_b, 3)]
        if min(g5_a, g5_b) < args.g5_side:
            warns.append('split r%d abandoned by G5 %s' % (rid, diag))
            continue
        new_id = freed.pop(0) if freed else len(rooms_out)
        if new_id >= len(rooms_out):
            rooms_out.append(None)
        rooms_out[rid] = {'pixel': ring_from_poly(a_piece_px)}
        rooms_out[new_id] = {'pixel': ring_from_poly(b_piece_px)}
        changed.update((rid, new_id))
        split_pieces[rid] = [(rid, a_piece_px), (new_id, b_piece_px)]
        id_map_ops[rid] = {'op': 'split', 'kept': rid, 'new': new_id,
                           'groups': op['groups'], 'cut': diag,
                           'cut_px': c_px}
        print('[refine] split r%d at ps %s=%.3f (col/row %d): kept r%d, '
              'new r%d' % (rid, 'xy'[axis], diag['c_final_ps'], c_px, rid,
                           new_id))
    for op in merge_ops:
        print('[refine] merged r%d into r%d' % (op['absorbed'],
                                                op['survivor']))
    assert all(r is not None for r in rooms_out), \
        'freed slots must all be refilled (unbalanced merge/split count?)'

    # ---- rebuild room dicts / json ----------------------------------------
    out_rooms = []
    for i, r in enumerate(rooms_out):
        if i not in changed:
            out_rooms.append(raw['rooms'][i])       # verbatim
            continue
        px = np.asarray(r['pixel'], dtype=np.float64)
        ps = pixel_to_ps(px[:, 0], px[:, 1], norm)
        out_rooms.append({'pixel': r['pixel'],
                          'world_mm': [[float(a), float(b)] for a, b in ps],
                          'world_m': [[float(a) / 1000., float(b) / 1000.]
                                      for a, b in ps]})
    data_out = {k: v for k, v in raw.items()}
    data_out['num_rooms'] = len(out_rooms)
    data_out['rooms'] = out_rooms
    data_out['openings'] = remap_openings(raw.get('openings', []),
                                          merge_map, split_pieces, warns)
    data_out['openings_undecided'] = remap_openings(
        raw.get('openings_undecided', []), merge_map, split_pieces, warns)
    data_out['align_info'] = dict(raw.get('align_info', {}))
    data_out['align_info']['refine'] = {
        'source': os.path.basename(cage_json),
        'infer_dir': args.infer_dir,
        'sparse_dir': sparse_dir,
        'thresholds': {k: getattr(args, k) for k in
                       ('t_link', 't_cross', 'g2_inside', 'g3_traj',
                        'g4_ovl', 'g5_side', 'snap_tol', 'cam_tol',
                        'h_band', 'micro_area',
                        'micro_edge', 'share_edge', 'spill_thr',
                        'self_cover', 'merge_traj')},
        'id_map': {str(k): v for k, v in sorted(id_map_ops.items())},
        'skipped': [ev for ev in split_ev
                    if 'insufficient' in ev.get('verdict', '')],
    }
    with open(out_json, 'w') as f:
        json.dump(data_out, f)
    print('[refine] refined CAGE json -> %s' % out_json)

    # readback checks
    chk = cc.load_cage(out_json)
    assert len(chk['rooms_xz']) == len(out_rooms)
    for i in range(len(out_rooms)):
        if i not in changed:
            assert raw['rooms'][i] is out_rooms[i]
    a_orig = sum(px_poly(r['pixel']).area for r in raw['rooms'])
    a_new = sum(px_poly(r['pixel']).area for r in out_rooms)
    assert abs(a_orig - a_new) < 1e-6, 'total pixel area changed'
    print('[refine] readback OK: %d rooms, total pixel area conserved (%.0f)'
          % (len(out_rooms), a_new))

    # ---- refined selection (frames re-assigned) ---------------------------
    os.makedirs(out_dir, exist_ok=True)
    polys_ps_new = [to_poly(r['world_mm']) for r in out_rooms]
    reassigned = {}
    for name, ring in rings_ps.items():
        old_rid = int(FRAME_RE.match(name).group(1))
        cand = old_rid
        if old_rid in merge_map:
            cand = merge_map[old_rid]
        if old_rid in split_pieces:
            pos = Point(*cc.world_to_ps(
                np.asarray([frame_xz[name]]), yaw)[0])
            side = [nid for nid, piece_px in split_pieces[old_rid]
                    if to_poly(pixel_to_ps(
                        np.asarray(piece_px.exterior.coords[:-1])[:, 0],
                        np.asarray(piece_px.exterior.coords[:-1])[:, 1],
                        norm)).buffer(1e-6).contains(pos)]
            cand = side[0] if side else old_rid
        reassigned[name] = cand
    sel_out = {k: v for k, v in sel.items()}
    sel_out['cage_json'] = os.path.abspath(out_json)
    sel_out['refined_from'] = os.path.abspath(
        os.path.join(render_dir, 'selection.json'))
    old_rooms_by_id = {r['id']: r for r in sel['rooms']}
    old_frame_recs = {fr['frame']: fr for r in sel['rooms']
                      for fr in r['frames']}
    rooms_sel = []
    for rid in range(len(out_rooms)):
        names = sorted([n for n, r in reassigned.items() if r == rid])
        ps_ring = np.asarray(out_rooms[rid]['world_mm'])
        xz_ring = cc.ps_to_world(ps_ring, yaw, norm['up_axis'])[
            :, list(cc.plan_axes(norm['up_axis']))]
        cen = cc._poly_centroid(xz_ring)
        frames = []
        for n in names:
            fr = dict(old_frame_recs[n])
            fr['dist_centroid'] = round(float(np.hypot(
                fr['xz'][0] - cen[0], fr['xz'][1] - cen[1])), 2)
            fr['inside'] = bool(to_poly(xz_ring).buffer(1e-6).contains(
                Point(*fr['xz'])))
            fr['ring_group'] = frame_group.get(n)
            frames.append(fr)
        base = old_rooms_by_id.get(rid, {})
        rooms_sel.append({
            'id': rid,
            'centroid_xz': [round(float(cen[0]), 3),
                            round(float(cen[1]), 3)],
            'poly_xz': [[round(float(a), 4), round(float(b), 4)]
                        for a, b in xz_ring],
            'fallback_outside': (base.get('fallback_outside', False)
                                 if rid not in changed else
                                 any(not f['inside'] for f in frames)),
            'no_coverage': len(frames) == 0,
            'n_centers': len({FRAME_RE.match(n).group(1, 2)
                              for n in names}),
            'frames': frames})
    sel_out['rooms'] = rooms_sel
    sel_out['num_frames'] = sum(len(r['frames']) for r in rooms_sel)
    assert sel_out['num_frames'] == len(rings_ps), 'frames lost in re-assign'
    sel_path = os.path.join(out_dir, 'refined_selection.json')
    with open(sel_path, 'w') as f:
        json.dump(sel_out, f, indent=2)
    moved = {n: (int(FRAME_RE.match(n).group(1)), r)
             for n, r in reassigned.items()
             if int(FRAME_RE.match(n).group(1)) != r}
    print('[refine] refined selection -> %s (%d frames moved: %s)'
          % (sel_path, len(moved),
             ', '.join('%s r%d->r%d' % (n, a, b)
                       for n, (a, b) in sorted(moved.items()))))

    # ---- report + overlay -------------------------------------------------
    report = {'scene': scene, 'cage_json': os.path.abspath(cage_json),
              'refined_json': os.path.abspath(out_json),
              'traj_poses_per_room': {str(k): v
                                      for k, v in sorted(traj_n.items())},
              'split_evidence': split_ev, 'merge_evidence': merge_ev,
              'operations': {str(k): v for k, v in sorted(id_map_ops.items())},
              'frames_moved': {n: {'from': a, 'to': b}
                               for n, (a, b) in sorted(moved.items())},
              'warnings': warns}
    with open(os.path.join(out_dir, 'refine_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print('[refine] report -> %s/refine_report.json' % out_dir)
    if warns:
        print('[refine] WARNINGS:\n  ' + '\n  '.join(warns))

    render_refine_overlay(cage, chk, room_rings, split_ev,
                          split_pieces, yaw, norm,
                          os.path.join(out_dir, 'refine_overlay.png'))


def render_refine_overlay(cage, cage_new, room_rings, split_ev,
                          split_pieces, yaw, norm, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm

    def ps_ring_to_world(ring_xy):
        return cc.ps_to_world(np.asarray(ring_xy), yaw, norm['up_axis'])[
            :, list(cc.plan_axes(norm['up_axis']))]

    fig, axes = plt.subplots(1, 2, figsize=(22, 11))
    group_cmap = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple']
    for ax, cg, title in ((axes[0], cage, 'original CAGE + LGT ring groups'),
                          (axes[1], cage_new, 'refined CAGE')):
        colors = [cm.tab20(i % 20) for i in range(len(cg['rooms_xz']))]
        for rid, xz in enumerate(cg['rooms_xz']):
            c = colors[rid]
            ax.fill(*np.vstack([xz, xz[:1]]).T, facecolor=c, alpha=0.18)
            ax.plot(*np.vstack([xz, xz[:1]]).T, color=c, lw=2.0)
            cen = cg['centroids'][rid]
            ax.text(cen[0], cen[1], str(rid), fontsize=14, fontweight='bold',
                    ha='center', va='center')
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(True, color='0.92')
    # left: every frame ring, coloured by the group its room split into
    ev_by_room = {e['room']: e for e in split_ev if 'groups' in e}
    for rid, rings in room_rings.items():
        groups = ev_by_room.get(rid, {}).get('groups') or [
            [n.split('_', 1)[1] for n in rings]]
        for gi, shorts in enumerate(groups):
            col = group_cmap[gi % len(group_cmap)]
            for s in shorts:
                ring = rings.get('r%02d_%s' % (rid, s))
                if ring is None:
                    continue
                for g in getattr(ring, 'geoms', [ring]):
                    w = ps_ring_to_world(np.asarray(g.exterior.coords))
                    axes[0].plot(w[:, 0], w[:, 1], color=col, lw=1.2,
                                 alpha=0.85)
    # right: highlight the new wall (shared edge of the two split pieces)
    for old_id, pieces in split_pieces.items():
        seg = pieces[0][1].boundary.intersection(pieces[1][1].boundary)
        geoms = getattr(seg, 'geoms', [seg])
        for g in geoms:
            if g.is_empty or not hasattr(g, 'coords'):
                continue
            v = np.asarray(g.coords)
            w = ps_ring_to_world(pixel_to_ps(v[:, 0], v[:, 1], norm))
            axes[1].plot(w[:, 0], w[:, 1], color='red', lw=4.0, alpha=0.9,
                         zorder=8, label='new wall (r%d split)' % old_id)
    if split_pieces:
        axes[1].legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print('[refine] overlay -> %s' % out_path)


if __name__ == '__main__':
    main()
