'''BEV comparison: LGT-Net per-pano layouts vs the CAGE floor plan (Method B).

Ported from uLayout/cage_postprocess/cage_bev_compare.py, adapted to consume
LGT-Net predictions (infer_gsplat_panos.py npz files) instead of uLayout
boundary curves. Virtual gsplat panos only ("exact" pose model): the pano
heading theta is known by construction, no calibration, no per-frame yaw
refinement.

Coordinate chain (see README.md for the derivation):
  LGT-Net predicts on the VP-LEVELED pano, in its model frame (x right,
  y down, z front = centre column), normalized so the floor plane is y=+1.
  For each corner p_model:

    V      = vp[2::-1]              rows of {name}_vp.txt, reversed
    M      = [[1,0,0],[0,0,-1],[0,1,0]]   pano_lsd frame -> model frame
    R_undo = M @ V.T @ M.T          undo the VP leveling rotation
    X_w    = R_c2w @ (h_cam * R_undo @ p_model) + C

  with R_c2w = poses[name].R^T = Ry(theta), C = camera_center, and
  h_cam = floor_y - C[1] (true gaussian floor; NEVER writer.py's 1.6 m).
  The BEV polygon is X_w[:, [0, 2]].

Per room the frame polygons are unioned and scored against the CAGE polygon
(IoU / coverage / overflow, overflow attributed to neighbour rooms), same
metrics as the uLayout version. Outputs bev_metrics.json + bev_overlay.png.

Run inside the LGT-Net conda env from the repo root:
    python cage_postprocess/cage_bev_compare.py \
        --infer_dir src/output/huizhongbeili-106/gsplat_infer_mp3d
'''
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, _ROOT)

import cage_common as cc

# pano_lsd frame (x lateral, y centre-column forward, z up) -> model frame
# (x right, y down, z forward). See utils/conversion.py header vs
# preprocessing/pano_lsd_align.py uv2xyzN(planeID=1).
M_LSD2MODEL = np.array([[1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0],
                        [0.0, 1.0, 0.0]])


def undo_vp_rotation(vp):
    '''Rotation (model frame) that maps leveled-pano points back to the
    original-pano camera frame, plus diagnostics.

    preprocess applies rotatePanorama(img, vp[2::-1]) with R = inv(V.T),
    V = vp[2::-1]: scene points transform p_aligned = R @ p_orig in the
    pano_lsd frame, so the undo is V.T (the exact inverse of R even when the
    6-decimal vp.txt rows are slightly non-orthogonal).
    '''
    V = np.asarray(vp, dtype=np.float64)[2::-1]
    det = float(np.linalg.det(V))
    if det <= 0:
        raise RuntimeError('det(V) = %.4f <= 0: mirrored VP frame, the '
                           'leveling would have flipped the pano' % det)
    R_undo = M_LSD2MODEL @ V.T @ M_LSD2MODEL.T
    # residual rotation angle of R_undo (identity for a perfectly level pano)
    cos_a = (np.trace(R_undo) - 1.0) / 2.0
    angle = float(np.rad2deg(np.arccos(np.clip(cos_a, -1.0, 1.0))))
    vert = float(abs(np.asarray(vp, dtype=np.float64)[0] @ [0.0, 0.0, 1.0]))
    return R_undo, angle, vert


def corners_world_xz(p_model, R_undo, R_c2w, C, h_cam, cap):
    '''Model-frame corners [N,3] (floor plane y=+1) -> world plan coords [N,2].'''
    p = np.asarray(p_model, dtype=np.float64).copy()
    r = np.linalg.norm(p[:, [0, 2]], axis=1)
    s = np.minimum(r, cap / max(h_cam, 1e-9)) / np.maximum(r, 1e-9)
    p[:, 0] *= s
    p[:, 2] *= s
    Xw = (R_c2w @ (h_cam * (R_undo @ p.T))).T + C
    return Xw[:, [0, 2]]


def to_poly(xz):
    from shapely.geometry import Polygon
    p = Polygon(np.round(np.asarray(xz, dtype=np.float64), 4))
    if not p.is_valid:
        p = p.buffer(0)
    return p


def room_metrics(union_ring, cage_polys, rid):
    '''IoU / overflow / coverage of one room's union vs its CAGE polygon.
    (identical to the uLayout version)'''
    cage = cage_polys[rid]
    inter = union_ring.intersection(cage).area
    union = union_ring.union(cage).area
    iou = inter / union if union > 0 else 0.0
    overflow_geom = union_ring.difference(cage)
    overflow = overflow_geom.area / union_ring.area if union_ring.area > 0 else 0.0
    coverage = inter / cage.area if cage.area > 0 else 0.0
    into = []
    for sid, other in enumerate(cage_polys):
        if sid == rid:
            continue
        a = overflow_geom.intersection(other).area
        if a > 0.05:
            into.append((sid, a))
    into.sort(key=lambda t: -t[1])
    outside_all = overflow_geom.area - sum(a for _, a in into)
    return {'iou': round(iou, 3), 'overflow': round(overflow, 3),
            'coverage': round(coverage, 3),
            'ring_area_m2': round(union_ring.area, 2),
            'cage_area_m2': round(cage.area, 2),
            'overflow_into': [{'room': sid, 'area_m2': round(a, 2)}
                              for sid, a in into[:3]],
            'overflow_outside_m2': round(max(outside_all, 0.0), 2)}


def render_overlay(cage, sel, ring_polys, cams_all, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm

    colors = [cm.tab20(i % 20) for i in range(len(cage['rooms_xz']))]
    fig, ax = plt.subplots(figsize=(14, 14))
    if len(cams_all):
        ax.plot(cams_all[:, 0], cams_all[:, 1], '.', color='0.85', ms=2,
                zorder=1, label='cameras')
    for rid, xz in enumerate(cage['rooms_xz']):
        c = colors[rid]
        ax.fill(*np.vstack([xz, xz[:1]]).T, facecolor=c, alpha=0.18, zorder=2)
        ax.plot(*np.vstack([xz, xz[:1]]).T, color=c, lw=2.2, zorder=3)
        cen = cage['centroids'][rid]
        ax.text(cen[0], cen[1], str(rid), color='k', fontsize=15,
                fontweight='bold', ha='center', va='center', zorder=6)
    for room in sel['rooms']:
        rid = room['id']
        c = colors[rid]
        for fr in room['frames']:
            key = (rid, fr['frame'])
            if key not in ring_polys:
                continue
            for xz in ring_polys[key]:
                ax.plot(*np.vstack([xz, xz[:1]]).T, color=c, lw=1.0,
                        alpha=0.9, zorder=4)
            ax.plot(fr['xz'][0], fr['xz'][1], 'o', color=c, ms=5, mec='k',
                    mew=0.5, zorder=5)
            ax.annotate(fr['frame'], fr['xz'], fontsize=6, color='0.3',
                        zorder=6, xytext=(3, 3), textcoords='offset points')
    ax.set_aspect('equal')
    ax.invert_yaxis()          # top-down view: match the CAGE floorplan PNGs
    ax.set_xlabel('world X (m)')
    ax.set_ylabel('world Z (m)')
    ax.set_title('CAGE rooms (filled) vs LGT-Net layouts (thin), top-down')
    ax.grid(True, color='0.92', zorder=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print('[bev] overlay -> %s' % out_path)


def debug_project(name, infer_dir, R_undo, R_c2w, C, h_cam, cage, sel, ddir,
                  out_dir):
    '''Decisive convention check: project the frame's own CAGE room polygon
    through the FORWARD chain onto the leveled pano and draw the wall lines.
    They must coincide with the visible wall/floor junctions.'''
    from PIL import Image
    from utils.conversion import xyz2pixel

    rid = next(r['id'] for r in sel['rooms']
               for fr in r['frames'] if fr['frame'] == name)
    poly = np.asarray(cage['rooms_xz'][rid], dtype=np.float64)
    # densify polygon edges for smooth curves on the pano
    pts = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        for t in np.linspace(0, 1, 64, endpoint=False):
            pts.append(a + (b - a) * t)
    pts = np.asarray(pts)
    meta = json.load(open(os.path.join(
        ddir, 'gsplat_render', 'virtual_camera_poses.json')))['metadata']
    img = np.array(Image.open(
        os.path.join(ddir, 'gsplat_render', 'renders', name + '.png'))
        .resize((1024, 512), Image.Resampling.BICUBIC))[..., :3].copy()
    # the model saw the LEVELED pano; re-level with the SAME cached vp
    from inference import preprocess
    img, _ = preprocess(img, vp_cache_path=os.path.join(
        infer_dir, '%s_vp.txt' % name))
    img = np.clip(img, 0, 255).astype(np.uint8)

    R_fwd = R_undo.T                      # model orig -> aligned
    for wy, color in ((meta['floor_y'], (255, 60, 60)),
                      (meta['ceil_y'], (60, 160, 255))):
        Xw = np.stack([pts[:, 0], np.full(len(pts), wy), pts[:, 1]], axis=1)
        p_cam = (R_c2w.T @ (Xw - C).T).T / h_cam
        p_al = (R_fwd @ p_cam.T).T
        uv = xyz2pixel(p_al, w=1024, h=512)
        for u, v in np.round(uv).astype(int):
            img[np.clip(v - 1, 0, 511):np.clip(v + 2, 0, 512),
                np.clip(u - 1, 0, 1023):np.clip(u + 2, 0, 1024)] = color
    out = os.path.join(out_dir, 'debug_project_%s.png' % name)
    Image.fromarray(img).save(out)
    print('[bev] debug projection (room %d walls onto leveled pano) -> %s'
          % (rid, out))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dataset_dir', default='src/datasets/huizhongbeili-106')
    ap.add_argument('--infer_dir', default=None,
                    help='infer_gsplat_panos.py output dir (default '
                         'src/output/<scene>/gsplat_infer_mp3d)')
    ap.add_argument('--out', default=None,
                    help='output dir (default <infer_dir>/../bev_compare_'
                         'gsplat_<tag from infer index.json>)')
    ap.add_argument('--curve', choices=['manhattan', 'raw'], default='manhattan',
                    help='manhattan corners (processed_xyz) or the raw '
                         '256-point boundary (raw_xyz)')
    ap.add_argument('--max-radius', type=float, default=12.0,
                    help='cap on corner distance from the camera (m)')
    ap.add_argument('--no-undo-vp', action='store_true',
                    help='ablation: skip the VP-leveling undo rotation')
    ap.add_argument('--debug-project', type=int, default=0, metavar='N',
                    help='project the CAGE room walls onto the first N leveled '
                         'panos (decisive convention check)')
    args = ap.parse_args()

    ddir = args.dataset_dir
    scene = os.path.basename(os.path.normpath(ddir))
    infer_dir = args.infer_dir or os.path.join('src', 'output', scene,
                                               'gsplat_infer_mp3d')
    index = json.load(open(os.path.join(infer_dir, 'index.json')))
    tag = index.get('tag', 'mp3d')
    out_dir = args.out or os.path.join(os.path.dirname(infer_dir.rstrip('/')),
                                       'bev_compare_gsplat_%s' % tag)
    os.makedirs(out_dir, exist_ok=True)

    sel = json.load(open(os.path.join(ddir, 'gsplat_render', 'selection.json')))
    cage_json = os.path.join(ddir, 'floorplan',
                             os.path.basename(sel['cage_json']))
    cage = cc.load_cage(cage_json)
    pdata = json.load(open(os.path.join(ddir, 'gsplat_render',
                                        'virtual_camera_poses.json')))
    poses, meta = pdata['poses'], pdata['metadata']
    cam_meta = json.load(open(os.path.join(ddir, 'gsplat_render',
                                           'cameras.json')))['meta']
    theta = float(meta['theta_deg'])
    assert abs(theta - float(cam_meta['theta_deg'])) < 1e-6, \
        'theta mismatch between poses metadata and cameras.json'
    floor_y, ceil_y = float(meta['floor_y']), float(meta['ceil_y'])
    print('[bev] scene %s, %d rooms, %d frames, theta=%.2f deg, model tag %s'
          % (scene, len(cage['rooms_xz']), len(index['frames']), theta, tag))

    cage_polys = [to_poly(xz) for xz in cage['rooms_xz']]
    key_xyz = 'processed_xyz' if args.curve == 'manhattan' else 'raw_xyz'

    ring_polys, metrics = {}, []
    angles, vert_dots, ceil_errs = [], [], []
    n_cam_outside = 0
    debug_left = args.debug_project
    for room in sel['rooms']:
        rid = room['id']
        shp = []
        for fr in room['frames']:
            name = fr['frame']
            npz_path = os.path.join(infer_dir, '%s_layout.npz' % name)
            if name not in poses or not os.path.isfile(npz_path):
                print('[bev]   room %d: missing pose/npz for %s' % (rid, name))
                continue
            z = np.load(npz_path)
            R_undo, angle, vert = undo_vp_rotation(z['vp'])
            if args.no_undo_vp:
                R_undo = np.eye(3)
            angles.append(angle)
            vert_dots.append(vert)
            if angle > 5.0:
                print('[bev]   warn: %s residual VP rotation %.1f deg'
                      % (name, angle))

            C = np.asarray(poses[name]['camera_center'], dtype=np.float64)
            R_c2w = np.asarray(poses[name]['R'], dtype=np.float64).T
            h_cam = floor_y - C[1]
            ceil_errs.append(float(z['ratio']) * h_cam - (C[1] - ceil_y))

            xz = corners_world_xz(z[key_xyz], R_undo, R_c2w, C, h_cam,
                                  args.max_radius)
            p = to_poly(xz)
            from shapely.geometry import Point
            if not p.contains(Point(C[0], C[2])):
                n_cam_outside += 1
            ring_polys[(rid, name)] = [xz]
            shp.append(p)

            if debug_left > 0:
                debug_project(name, infer_dir, R_undo, R_c2w, C, h_cam, cage,
                              sel, ddir, out_dir)
                debug_left -= 1
        if not shp:
            metrics.append({'room': rid, 'error': 'no polygons'})
            continue
        from shapely.ops import unary_union
        union_ring = unary_union(shp)
        m = {'room': rid,
             'frames': [fr['frame'] for fr in room['frames']],
             'frame_iou': [round(cage_polys[rid].intersection(p).area /
                                 max(cage_polys[rid].union(p).area, 1e-9), 3)
                           for p in shp]}
        m.update(room_metrics(union_ring, cage_polys, rid))
        metrics.append(m)

    print('[bev] VP residual rotation: max %.2f deg, mean %.2f deg; '
          '|vp0.z|: min %.4f' % (max(angles), float(np.mean(angles)),
                                 min(vert_dots)))
    true_hc = float(next(iter(poses.values()))['camera_center'][1]) - ceil_y
    print('[bev] ceiling check (ratio*h_cam - true %.3f m): '
          'mean %+0.3f m, max|err| %.3f m'
          % (true_hc, float(np.mean(ceil_errs)),
             float(np.max(np.abs(ceil_errs)))))
    if n_cam_outside:
        print('[bev] warn: %d frame polygons do not contain their own camera'
              % n_cam_outside)

    print('\n[bev] room  IoU  cover overflow  pred/cage m2   overflow into')
    for m in metrics:
        if 'error' in m:
            print('  %4d  %s' % (m['room'], m['error']))
            continue
        into = ' '.join('r%d:%.1f' % (d['room'], d['area_m2'])
                        for d in m['overflow_into'])
        if m['overflow_outside_m2'] > 0.3:
            into += ' out:%.1f' % m['overflow_outside_m2']
        print('  %4d  %.2f  %.2f   %.2f     %5.1f/%5.1f    %s'
              % (m['room'], m['iou'], m['coverage'], m['overflow'],
                 m['ring_area_m2'], m['cage_area_m2'], into))
    ious = [m['iou'] for m in metrics if 'iou' in m]
    print('[bev] mean IoU %.3f over %d rooms' % (float(np.mean(ious)),
                                                 len(ious)))

    out_json = os.path.join(out_dir, 'bev_metrics.json')
    with open(out_json, 'w') as f:
        json.dump({'scene': scene, 'curve': args.curve, 'model_tag': tag,
                   'theta_deg': theta, 'undo_vp': not args.no_undo_vp,
                   'vp_residual_deg': {'max': round(max(angles), 3),
                                       'mean': round(float(np.mean(angles)), 3)},
                   'ceiling_err_m': {'mean': round(float(np.mean(ceil_errs)), 4),
                                     'max_abs': round(float(np.max(
                                         np.abs(ceil_errs))), 4)},
                   'mean_iou': round(float(np.mean(ious)), 3),
                   'rooms': metrics}, f, indent=2)
    print('[bev] metrics -> %s' % out_json)

    cams_all = np.array([[p['camera_center'][0], p['camera_center'][2]]
                         for p in poses.values()])
    render_overlay(cage, sel, ring_polys, cams_all,
                   os.path.join(out_dir, 'bev_overlay.png'))


if __name__ == '__main__':
    main()
