'''Generate ideal virtual pano cameras for gsplat360 from the CAGE floor plan.

Ported from uLayout/cage_postprocess/gen_gsplat_cameras.py (Method B, step 1).
Differences from the uLayout original:
  - self-contained (find_cage_json / points3D.bin parsing inlined, no
    3d_layout_viewer dependency);
  - default --dataset_dir points at LGT-Net's src/datasets/<scene>/;
  - new --verify mode: regenerate everything in memory and compare against the
    gsplat_render/{cameras,virtual_camera_poses}.json copied from uLayout
    (hard-check theta/camera_y/floor_y/ceil_y/R/qvec and the view name set;
    per-view sampled xz is soft-checked only, shapely version differences may
    reorder representative/jitter points). Verify mode never writes files.

Pipeline:
  1. verify all three data sources share one COLMAP world: CAGE json vs
     sparse/0 (cage_common.check_same_world) and the trained gaussians in
     point_cloud.ply vs sparse/0 (check_gaussian_world);
  2. sample viewpoints per CAGE room: shrink the polygon by a wall clearance,
     split large rooms into ~9 m2 axis-aligned subregions, and take 3 points
     (centre + 2 small jitters) per subregion centre;
  3. build one shared level camera rotation aligned with the CAGE plan axis:
     camtoworld = Ry(theta) (OpenCV cols, zero pitch/roll, det=+1);
  4. write <dataset_dir>/gsplat_render/{cameras.json, virtual_camera_poses.json,
     selection.json, viewpoints.png}.

Run inside the LGT-Net conda env from cage_postprocess/:
    python gen_gsplat_cameras.py --dataset_dir ../src/datasets/huizhongbeili-106 --verify
'''
import argparse
import glob
import json
import os
import re
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cage_common as cc

PANO_W, PANO_H = 2048, 1024


def find_cage_json(ddir):
    hits = glob.glob(os.path.join(ddir, 'floorplan', '*_aligned_polys.json'))
    if len(hits) != 1:
        raise RuntimeError('expected exactly one *_aligned_polys.json under '
                           '%s/floorplan, found %d' % (ddir, len(hits)))
    return hits[0]


def read_points3d_bin(path):
    '''Return xyz[N,3] float32 from a COLMAP points3D.bin.'''
    xyz = []
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            f.read(8)                                # point id
            xyz.append(struct.unpack('<3d', f.read(24)))
            f.read(3 + 8)                            # rgb + reprojection error
            track_len = struct.unpack('<Q', f.read(8))[0]
            f.read(track_len * 8)
    return np.asarray(xyz, dtype=np.float32)


def load_points3d_cached(points3d_bin, cache_path):
    '''xyz from points3D.bin with npz cache (compatible with uLayout's cache).'''
    if os.path.isfile(cache_path):
        return np.load(cache_path)['xyz']
    print('[gsplat-cam] parsing %s (one-off, building cache) ...' % points3d_bin)
    xyz = read_points3d_bin(points3d_bin)
    np.savez(cache_path, xyz=xyz)
    print('[gsplat-cam] cached %d points -> %s' % (len(xyz), cache_path))
    return xyz


def check_gaussian_world(ply_path, sparse_xyz, inside_min=0.90,
                         scale_tol=0.05, shift_tol=0.5, stride=20):
    '''Verify the 3DGS point_cloud.ply lives in the sparse/0 COLMAP world.

    3DGS clouds carry far-away floater gaussians (sky through windows etc.),
    so global percentiles are meaningless. Gate instead on (a) the fraction of
    gaussian centres inside the padded sparse bbox and (b) plan-axis span
    ratio / midpoint shift of the in-bbox subset vs the sparse percentiles.
    '''
    with open(ply_path, 'rb') as f:
        header = b''
        while not header.endswith(b'end_header\n'):
            header += f.readline()
        off = f.tell()
    n = int(re.search(rb'element vertex (\d+)', header).group(1))
    props = re.findall(rb'property float (\w+)', header)
    names = [p.decode() for p in props]
    if names[:3] != ['x', 'y', 'z']:
        raise RuntimeError('%s: unexpected ply layout %s' % (ply_path, names[:3]))
    stride_b = 4 * len(names)
    mm = np.memmap(ply_path, dtype=np.uint8, mode='r', offset=off)
    idx = np.arange(0, n, stride)
    xyz = np.empty((len(idx), 3), np.float32)
    for k, i in enumerate(idx):
        xyz[k] = np.frombuffer(mm[i * stride_b:i * stride_b + 12].tobytes(),
                               np.float32)

    slo = np.percentile(sparse_xyz, 1, axis=0)
    shi = np.percentile(sparse_xyz, 99, axis=0)
    inside = np.all((xyz > slo - 0.5) & (xyz < shi + 0.5), axis=1)
    frac = float(inside.mean())
    gin = xyz[inside]
    g2, g98 = np.percentile(gin, 2, 0), np.percentile(gin, 98, 0)
    s2, s98 = np.percentile(sparse_xyz, 2, 0), np.percentile(sparse_xyz, 98, 0)
    ratio = (g98 - g2) / (s98 - s2)
    shift = ((g2 + g98) - (s2 + s98)) / 2.0
    print('[gsplat-cam] gaussian-vs-sparse world check (%d/%d sampled centres):'
          % (len(xyz), n))
    print('  inside sparse bbox+0.5m: %.1f%% (floaters outside)' % (100 * frac))
    print('  in-bbox span ratio (%.3f %.3f %.3f)  midpoint shift '
          '(%+.3f %+.3f %+.3f) m' % (tuple(ratio) + tuple(shift)))
    bad_scale = float(np.abs(ratio[[0, 2]] - 1.0).max())
    bad_shift = float(np.abs(shift[[0, 2]]).max())
    if frac < inside_min or bad_scale > scale_tol or bad_shift > shift_tol:
        raise RuntimeError(
            'point_cloud.ply does not match the sparse/0 world (inside %.1f%%, '
            'scale off %.1f%%, shift %.2f m) -- was the 3DGS trained with '
            'normalize=True? Realign before rendering.'
            % (100 * frac, 100 * bad_scale, bad_shift))
    print('  -> same world (inside %.1f%% >= %.0f%%, plan scale within %.1f%%, '
          'shift %.2f m)' % (100 * frac, 100 * inside_min, 100 * bad_scale,
                             bad_shift))


def measure_gaussian_planes(ply_path, region_xz, margin=0.5, bin_m=0.02):
    '''Floor / ceiling planes of the 3DGS cloud as its dominant horizontal
    gaussian slabs (world up is -Y: floor at the +Y extreme, ceiling at -Y).'''
    with open(ply_path, 'rb') as f:
        header = b''
        while not header.endswith(b'end_header\n'):
            header += f.readline()
        off = f.tell()
    n = int(re.search(rb'element vertex (\d+)', header).group(1))
    names = [p.decode() for p in re.findall(rb'property float (\w+)', header)]
    col = {m: i for i, m in enumerate(names)}
    mm = np.memmap(ply_path, dtype=np.float32, mode='r', offset=off,
                   shape=(n, len(names)))
    x = np.asarray(mm[:, col['x']])
    y = np.asarray(mm[:, col['y']])
    z = np.asarray(mm[:, col['z']])
    op = np.asarray(mm[:, col['opacity']])          # logit; >0 == sigmoid>0.5
    x0, z0, x1, z1 = region_xz
    keep = ((op > 0.0) & (x > x0 - margin) & (x < x1 + margin)
            & (z > z0 - margin) & (z < z1 + margin))
    yy = y[keep]
    lo, hi = np.percentile(yy, 2), np.percentile(yy, 98)
    edges = np.arange(lo - 0.3, hi + 0.3, bin_m)
    hist, _ = np.histogram(yy, bins=edges)
    cen = (edges[:-1] + edges[1:]) / 2.0
    floor_y = float(cen[np.argmax(np.where(cen > hi - 0.5, hist, -1))])
    ceil_y = float(cen[np.argmax(np.where(cen < lo + 0.5, hist, -1))])
    print('[gsplat-cam] gaussian planes (%d central centres): floor y=%.3f '
          '(+Y/down), ceil y=%.3f (-Y/up), room height %.2f m'
          % (int(keep.sum()), floor_y, ceil_y, floor_y - ceil_y))
    return floor_y, ceil_y


def plan_axis_theta(cage):
    '''World-plan heading (deg) of the CAGE wall axis nearest to world +Z.'''
    d = cc.undo_yaw(np.array([[1.0, 0.0]]), cage['yaw'])[0]
    base = np.rad2deg(np.arctan2(d[0], d[1]))
    cands = [(base + k * 90.0 + 180.0) % 360.0 - 180.0 for k in range(4)]
    return min(cands, key=abs)


def cam_to_world(theta_deg):
    '''Shared level camtoworld rotation = Ry(theta). Columns (camera axes in
    world): x=right, y=(0,1,0) [image-down = world +Y = gravity down], z=pano
    centre-column heading. Proper (det=+1), zero pitch/roll.'''
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_to_qvec(R):
    '''3x3 rotation -> COLMAP-style quaternion (w, x, y, z).'''
    t = np.trace(R)
    if t > 0:
        w = np.sqrt(1.0 + t) / 2.0
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
        q = [0.0, 0.0, 0.0]
        q[i] = s / 4.0
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        w = (R[k, j] - R[j, k]) / s
        x, y, z = q
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def shrink_room(poly, clearances=(0.3, 0.15)):
    for c in clearances:
        s = poly.buffer(-c)
        if not s.is_empty and s.area > 1e-3:
            return s, c
    return poly, 0.0


def subregion_centers(shrunk, theta_deg, split, cell_target):
    '''Centres covering the room: 1 for small rooms, an axis-aligned grid of
    ~cell_target m2 cells (in the CAGE-axis frame) for large ones.'''
    from shapely.affinity import rotate
    from shapely.geometry import box
    if not split:
        return [shrunk.representative_point().coords[0]]
    ax = rotate(shrunk, -theta_deg, origin=(0, 0))
    x0, z0, x1, z1 = ax.bounds
    n = max(int(np.ceil(ax.area / cell_target)), 2)
    side = float(np.sqrt(ax.area / n))
    nx = max(int(round((x1 - x0) / side)), 1)
    nz = max(int(round((z1 - z0) / side)), 1)
    centers = []
    for i in range(nx):
        for j in range(nz):
            cell = box(x0 + (x1 - x0) * i / nx, z0 + (z1 - z0) * j / nz,
                       x0 + (x1 - x0) * (i + 1) / nx,
                       z0 + (z1 - z0) * (j + 1) / nz)
            inter = ax.intersection(cell)
            if inter.area >= 1.5:
                p = inter.representative_point()
                back = rotate(p, theta_deg, origin=(0, 0))
                centers.append(back.coords[0])
    return centers or [shrunk.representative_point().coords[0]]


def sample_points(shrunk, center, n_samples, jitter, rng):
    from shapely.geometry import Point
    pts = [tuple(center)]
    tries = 0
    while len(pts) < n_samples and tries < 30:
        a = rng.uniform(0, 2 * np.pi)
        r = jitter if tries < 15 else jitter / 2
        p = (center[0] + r * np.cos(a), center[1] + r * np.sin(a))
        tries += 1
        if shrunk.contains(Point(*p)):
            pts.append(p)
    while len(pts) < n_samples:      # tiny room: duplicates are harmless
        pts.append(tuple(center))
    return pts


def render_viewpoints_png(cage, rooms_out, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    fig, ax = plt.subplots(figsize=(11, 11))
    for rid, xz in enumerate(cage['rooms_xz']):
        c = cm.tab20(rid % 20)
        ax.fill(*np.vstack([xz, xz[:1]]).T, facecolor=c, alpha=0.15, zorder=1)
        ax.plot(*np.vstack([xz, xz[:1]]).T, color=c, lw=1.8, zorder=2)
        cen = cage['centroids'][rid]
        ax.text(cen[0], cen[1], str(rid), fontsize=13, fontweight='bold',
                ha='center', va='center', zorder=5)
    for room in rooms_out:
        c = cm.tab20(room['id'] % 20)
        for fr in room['frames']:
            ax.plot(fr['xz'][0], fr['xz'][1], 'o', color=c, ms=4, mec='k',
                    mew=0.4, zorder=4)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlabel('world X (m)')
    ax.set_ylabel('world Z (m)')
    ax.set_title('virtual pano viewpoints on CAGE rooms (top-down)')
    ax.grid(True, color='0.92', zorder=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('[gsplat-cam] viewpoints -> %s' % path)


def build_all(args):
    '''Run checks + sampling + pose construction; return everything in memory.'''
    ddir = args.dataset_dir
    cage_json = args.cage_json or find_cage_json(ddir)
    cage = cc.load_cage(cage_json)
    print('[gsplat-cam] CAGE json: %s (%d rooms, yaw %.1f deg)'
          % (cage_json, len(cage['rooms_xz']), cage['yaw']))

    sparse_dir = os.path.join(ddir, 'sparse', '0')
    xyz = load_points3d_cached(
        os.path.join(sparse_dir, 'points3D.bin'),
        os.path.join(sparse_dir, 'points3d_cache.npz'))
    cc.check_same_world(xyz, cage['norm'])
    ply_path = os.path.join(ddir, 'point_cloud.ply')
    check_gaussian_world(ply_path, xyz, inside_min=args.inside_min)

    all_xz = np.vstack(cage['rooms_xz'])
    region_xz = (all_xz[:, 0].min(), all_xz[:, 1].min(),
                 all_xz[:, 0].max(), all_xz[:, 1].max())
    floor_y, ceil_y = measure_gaussian_planes(ply_path, region_xz)

    theta = plan_axis_theta(cage)
    R_c2w = cam_to_world(theta)
    cam_y = floor_y - args.height          # up is -Y: subtract to rise
    print('[gsplat-cam] pano heading theta=%.2f deg (CAGE wall axis), camera '
          'y=%.3f (gaussian floor %.3f - %.2f up; CAGE floor %.3f for reference)'
          % (theta, cam_y, floor_y, args.height, cage['y_floor']))

    from shapely.geometry import Polygon
    rng = np.random.default_rng(0)
    views, rooms_out = [], []
    for rid, xz in enumerate(cage['rooms_xz']):
        poly = Polygon(xz).buffer(0)
        shrunk, clr = shrink_room(poly)
        centers = subregion_centers(shrunk, theta,
                                    poly.area > args.split_area,
                                    args.cell_area)
        frames = []
        for cid, cen in enumerate(centers):
            for sid, p in enumerate(sample_points(shrunk, cen,
                                                  args.samples_per_center,
                                                  args.jitter, rng)):
                name = 'r%02d_c%d_s%d' % (rid, cid, sid)
                views.append({'name': name, 'room_id': rid, 'center_id': cid,
                              'xz': [round(p[0], 3), round(p[1], 3)]})
                d = float(np.hypot(p[0] - cage['centroids'][rid][0],
                                   p[1] - cage['centroids'][rid][1]))
                frames.append({'frame': name,
                               'xz': [round(p[0], 3), round(p[1], 3)],
                               'dist_centroid': round(d, 2), 'inside': True,
                               'h_floor': args.height})
        rooms_out.append({'id': rid,
                          'centroid_xz': [round(float(v), 3)
                                          for v in cage['centroids'][rid]],
                          'poly_xz': np.round(xz, 3).tolist(),
                          'fallback_outside': False,
                          'clearance': clr,
                          'n_centers': len(centers),
                          'frames': frames})
        print('  room %2d: area %5.1f m2 -> %d center(s), %d views'
              % (rid, poly.area, len(centers), len(frames)))

    return {'cage': cage, 'cage_json': cage_json, 'theta': theta,
            'R_c2w': R_c2w, 'cam_y': cam_y, 'floor_y': floor_y,
            'ceil_y': ceil_y, 'views': views, 'rooms_out': rooms_out}


def verify(built, ddir):
    '''Compare the in-memory result against the copied gsplat_render jsons.'''
    out_dir = os.path.join(ddir, 'gsplat_render')
    ref_cams = json.load(open(os.path.join(out_dir, 'cameras.json')))
    ref_poses = json.load(open(os.path.join(out_dir,
                                            'virtual_camera_poses.json')))
    meta = ref_cams['meta']
    ok = True

    def hard(name, got, want, tol):
        nonlocal ok
        err = abs(float(got) - float(want))
        good = err <= tol
        ok = ok and good
        print('  [%s] %-16s got %10.4f  ref %10.4f  |err| %.2e'
              % ('OK' if good else 'FAIL', name, got, want, err))

    print('[verify] scalar meta (tol 1e-3):')
    hard('theta_deg', built['theta'], meta['theta_deg'], 1e-3)
    hard('camera_y', built['cam_y'], meta['camera_y'], 1e-3)
    hard('floor_y', built['floor_y'], meta['floor_y'], 1e-3)
    hard('ceil_y', built['ceil_y'], meta['ceil_y'], 1e-3)

    R_w2c = built['R_c2w'].T
    qvec = rot_to_qvec(R_w2c)
    any_pose = next(iter(ref_poses['poses'].values()))
    dR = float(np.abs(R_w2c - np.asarray(any_pose['R'])).max())
    dq = float(np.abs(qvec - np.asarray(any_pose['qvec'])).max())
    good = dR <= 1e-6 and dq <= 1e-6
    ok = ok and good
    print('  [%s] shared R/qvec     max|dR| %.2e  max|dq| %.2e'
          % ('OK' if good else 'FAIL', dR, dq))

    names_new = {v['name'] for v in built['views']}
    names_ref = {v['name'] for v in ref_cams['views']}
    good = names_new == names_ref and len(built['views']) == len(ref_cams['views'])
    ok = ok and good
    print('  [%s] view set           %d generated vs %d reference%s'
          % ('OK' if good else 'FAIL', len(built['views']),
             len(ref_cams['views']),
             '' if good else ' (diff: %s)' % sorted(names_new ^ names_ref)[:6]))

    # soft check: sampled xz positions (shapely version differences may shift
    # representative_point / jitter acceptance -- report only, copied
    # cameras.json stays authoritative because the renders correspond to it)
    ref_xz = {v['name']: v['xz'] for v in ref_cams['views']}
    d = [float(np.hypot(v['xz'][0] - ref_xz[v['name']][0],
                        v['xz'][1] - ref_xz[v['name']][1]))
         for v in built['views'] if v['name'] in ref_xz]
    if d:
        print('  [soft] sampled xz deviation: max %.4f m, mean %.4f m '
              '(informative only)' % (max(d), float(np.mean(d))))

    if not ok:
        raise SystemExit('[verify] FAILED: generated cameras disagree with the '
                         'copied gsplat_render jsons')
    print('[verify] PASSED: same world, same shared pose, same view set')


def write_outputs(built, ddir, args):
    out_dir = os.path.join(ddir, 'gsplat_render')
    os.makedirs(out_dir, exist_ok=True)
    for fn in ('cameras.json', 'virtual_camera_poses.json', 'selection.json'):
        if os.path.exists(os.path.join(out_dir, fn)) and not args.force:
            raise SystemExit('%s exists; pass --force to overwrite (the '
                             'renders/ were made from the existing cameras)'
                             % os.path.join(out_dir, fn))

    theta, cam_y = built['theta'], built['cam_y']
    cameras = {'meta': {
        'world': 'COLMAP sparse/0 world (verified vs CAGE json and '
                 'point_cloud.ply)',
        'theta_deg': round(theta, 3),
        'camera_y': round(cam_y, 4),
        'height_above_floor': args.height,
        'floor_y': round(built['floor_y'], 4),
        'ceil_y': round(built['ceil_y'], 4),
        'width': PANO_W, 'height': PANO_H,
        'convention': 'world up is -Y (gravity +Y). camtoworld = Ry(theta) '
                      '(OpenCV cols [x,y,z], image-down = world +Y). This '
                      'upright camera renders ceiling-at-top / floor-at-bottom '
                      'directly, so render_gsplat_panos.py applies NO flip; the '
                      'pano frame -> world rotation is Ry(theta_deg).',
        'cage_json': os.path.abspath(built['cage_json']),
    }, 'views': []}
    poses = {}
    R_w2c = built['R_c2w'].T
    qvec = rot_to_qvec(R_w2c)
    for v in built['views']:
        c = np.array([v['xz'][0], cam_y, v['xz'][1]])
        tvec = -R_w2c @ c
        cameras['views'].append({
            'name': v['name'], 'room_id': v['room_id'],
            'center_id': v['center_id'], 'xz': v['xz'],
            'camtoworld': np.round(np.vstack([
                np.hstack([built['R_c2w'], c[:, None]]),
                [0, 0, 0, 1]]), 8).tolist()})
        poses[v['name']] = {'qvec': qvec.tolist(),
                            'tvec': np.round(tvec, 6).tolist(),
                            'R': np.round(R_w2c, 8).tolist(),
                            'camera_center': np.round(c, 6).tolist()}

    with open(os.path.join(out_dir, 'cameras.json'), 'w') as f:
        json.dump(cameras, f, indent=1)
    with open(os.path.join(out_dir, 'virtual_camera_poses.json'), 'w') as f:
        json.dump({'metadata': {'reference_camera': 'virtual gsplat pano',
                                'theta_deg': round(theta, 3),
                                'floor_y': round(built['floor_y'], 4),
                                'ceil_y': round(built['ceil_y'], 4),
                                'height_above_floor': args.height,
                                'note': 'ideal level virtual cameras '
                                        '(world->camera), metric world; '
                                        'floor_y/ceil_y are the true gaussian '
                                        'planes for ring-height reconstruction'},
                   'poses': poses}, f, indent=1)
    with open(os.path.join(out_dir, 'selection.json'), 'w') as f:
        json.dump({'cage_json': os.path.abspath(built['cage_json']),
                   'per_room': args.samples_per_center,
                   'num_frames': len(built['views']),
                   'virtual': True,
                   'rooms': built['rooms_out']}, f, indent=2)
    print('[gsplat-cam] %d views -> %s/{cameras,virtual_camera_poses,'
          'selection}.json' % (len(built['views']), out_dir))
    render_viewpoints_png(built['cage'], built['rooms_out'],
                          os.path.join(out_dir, 'viewpoints.png'))
    print('\nnext: copy point_cloud.ply + gsplat_render/cameras.json + '
          'render_gsplat_panos.py to the CUDA machine and run\n'
          '  python render_gsplat_panos.py --ply point_cloud.ply '
          '--cameras cameras.json --out renders/')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dataset_dir',
                    default='../src/datasets/huizhongbeili-106')
    ap.add_argument('--cage-json', default=None)
    ap.add_argument('--height', type=float, default=1.75,
                    help='camera height above the true (gaussian) floor (m, '
                         'toward world up = -Y)')
    ap.add_argument('--samples-per-center', type=int, default=3)
    ap.add_argument('--jitter', type=float, default=0.25)
    ap.add_argument('--split-area', type=float, default=12.0)
    ap.add_argument('--cell-area', type=float, default=9.0)
    ap.add_argument('--inside-min', type=float, default=0.90)
    ap.add_argument('--verify', action='store_true',
                    help='regenerate in memory and compare against the copied '
                         'gsplat_render jsons; never writes files')
    ap.add_argument('--force', action='store_true',
                    help='allow overwriting existing gsplat_render jsons')
    args = ap.parse_args()

    built = build_all(args)
    if args.verify:
        verify(built, args.dataset_dir)
    else:
        write_outputs(built, args.dataset_dir, args)


if __name__ == '__main__':
    main()
