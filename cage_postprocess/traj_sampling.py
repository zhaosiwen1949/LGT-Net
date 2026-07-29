'''Trajectory-driven viewpoint sampling for virtual gsplat panos.

Replaces the CAGE-polygon grid sampler with viewpoints taken from the REAL
camera trajectory in sparse/0 (COLMAP 3.11+ rig format, frames.bin):
  - real positions are guaranteed reachable (no viewpoints inside beds /
    wardrobes that the grid sampler produces);
  - trajectory frames outside every CAGE room expose rooms CAGE missed;
  - hierarchical clustering (per-room KMeans + per-cluster farthest-point
    picks, DBSCAN for outside points) balances coverage against redundancy.

Only the (x, z) of the real frames is used -- the rendered virtual cameras
keep the uniform height (1.75 m above the gaussian floor) and the uniform
level heading Ry(theta), so the downstream "exact" pose model is unchanged.

Selection logic mirrors uLayout cage_postprocess/select_room_panos.pick_frames
(nearest-to-anchor + greedy farthest-point sampling).
'''
import json
import os
import struct

import numpy as np

FRAME_REC_BYTES = 164    # frame_id+rig_id(8) + quat(32) + trans(24) + num_data_ids(8) + data_ids(92)


def _quat_to_R(w, x, y, z):
    '''COLMAP (w,x,y,z) quaternion -> world->camera rotation matrix.'''
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def load_trajectory(sparse_dir, cross_check_json=None):
    '''Parse sparse/0/frames.bin -> (frame_names, camera_centers[N,3]).

    COLMAP 3.11+ rig format: fixed 164-byte records of
      frame_id(u32) rig_id(u32) quat(4xf64 wxyz) trans(3xf64)
      num_data_ids(u64) data_ids(...)
    rig_from_world equals the pano_camera0 world->camera extrinsics (reference
    sensor), so C = -R^T @ t. Frame name = 'frame_%05d' % (frame_id - 1).
    '''
    path = os.path.join(sparse_dir, 'frames.bin')
    size = os.path.getsize(path)
    if (size - 8) % FRAME_REC_BYTES != 0:
        raise RuntimeError('%s: size %d does not match the 164-byte rig-frame '
                           'record layout' % (path, size))
    names, centers = [], []
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            rec = f.read(FRAME_REC_BYTES)
            frame_id, _rig_id = struct.unpack('<II', rec[:8])
            w, x, y, z = struct.unpack('<4d', rec[8:40])
            t = np.array(struct.unpack('<3d', rec[40:64]))
            R = _quat_to_R(w, x, y, z)
            names.append('frame_%05d' % (frame_id - 1))
            centers.append(-R.T @ t)
    centers = np.asarray(centers)
    if cross_check_json and os.path.isfile(cross_check_json):
        ref = json.load(open(cross_check_json))['poses']
        picked = [i for i in (0, len(names) // 2, len(names) - 1)
                  if names[i] in ref]
        for i in picked:
            d = float(np.abs(centers[i]
                             - np.asarray(ref[names[i]]['camera_center'])).max())
            if d > 1e-6:
                raise RuntimeError('frames.bin vs %s disagree on %s '
                                   '(max diff %.2e)'
                                   % (cross_check_json, names[i], d))
        print('[traj] cross-check vs %s: %d frames, diff <= 1e-6'
              % (os.path.basename(cross_check_json), len(picked)))
    print('[traj] %d trajectory frames from %s' % (len(names), path))
    return names, centers


def classify_frames(centers, room_polys, floor_y, h_band=(0.9, 1.9),
                    boundary_margin=0.15):
    '''Assign trajectory frames to CAGE rooms / outside.

    Returns dict:
      room_members       {rid: idx array}  frames inside room rid (any height)
      room_members_hok   {rid: idx array}  ... and within the height band
      outside_idx        idx array         height-band frames in NO room and
                                           farther than boundary_margin from
                                           every room boundary (wall-thickness /
                                           door-crossing noise removed)
      heights            [N] camera height above the true floor
    '''
    from shapely.geometry import Point

    heights = floor_y - centers[:, 1]          # world up is -Y
    h_ok = (heights >= h_band[0]) & (heights <= h_band[1])
    pts = [Point(p[0], p[2]) for p in centers]

    room_members, room_members_hok = {}, {}
    inside_any = np.zeros(len(centers), dtype=bool)
    for rid, poly in enumerate(room_polys):
        m = np.array([poly.contains(p) for p in pts])
        room_members[rid] = np.where(m)[0]
        room_members_hok[rid] = np.where(m & h_ok)[0]
        inside_any |= m

    out = np.where(~inside_any & h_ok)[0]
    if len(out):
        near_wall = np.array([
            min(poly.exterior.distance(pts[i]) for poly in room_polys)
            <= boundary_margin for i in out])
        out = out[~near_wall]
    return {'room_members': room_members, 'room_members_hok': room_members_hok,
            'outside_idx': out, 'heights': heights}


def greedy_pick(xz, cand, anchor, n):
    '''Pick n of cand: nearest to anchor first, then greedy farthest-point.

    xz: [N,2] all frame positions; cand: candidate indices into xz;
    anchor: [2] reference point (cluster centre). Mirrors uLayout
    select_room_panos.pick_frames.
    '''
    cand = list(cand)
    if not cand:
        return []
    d_anchor = np.linalg.norm(xz[cand] - anchor, axis=1)
    picked = [cand.pop(int(np.argmin(d_anchor)))]
    while cand and len(picked) < n:
        d = np.min(np.linalg.norm(
            xz[cand][:, None, :] - xz[picked][None, :, :], axis=2), axis=1)
        picked.append(cand.pop(int(np.argmax(d))))
    return picked


def sample_room(poly, xz, member_idx, cell_area=9.0, per_cluster=3,
                wall_clear=0.3, seed=0, min_clusters=1, min_cluster_area=2.0,
                openings=None, opening_clear=0.0):
    '''Cluster one room's trajectory frames and pick representatives.

    KMeans with k = min(max(ceil(room area / cell_area), floor), n_members) on
    the frame (x, z), where floor = min_clusters once the room is at least
    min_cluster_area. The floor exists because the area rule alone leaves every
    room under cell_area with a single cluster, and a single cluster carries no
    evidence for refine_cage_rooms' split test -- a 4 m2 CAGE room that merges
    two real rooms can never be detected. Cluster ids are re-numbered by centre
    coordinates for determinism.

    Within each cluster prefer frames that clear the CAGE walls by wall_clear
    (poly.buffer(-wall_clear)) AND stay opening_clear away from the CAGE
    door/passage segments (`openings`, world-plan [2,2] arrays) -- a camera
    standing in a doorway sees both rooms and predicts a layout spanning them.
    Clearance is relaxed (walls first, then openings) if nothing qualifies.
    Returns list of cluster dicts:
      {'center_id', 'center_xz', 'picks': [frame idx], 'clearance_relaxed',
       'opening_relaxed'}
    '''
    from shapely.geometry import LineString, Point
    from sklearn.cluster import KMeans

    member_idx = np.asarray(member_idx)
    if len(member_idx) == 0:
        return []
    pts = xz[member_idx]
    k = int(np.ceil(poly.area / cell_area))
    if poly.area >= min_cluster_area:
        k = max(k, min_clusters)
    k = max(1, min(k, len(member_idx)))
    if k == 1:
        labels = np.zeros(len(member_idx), dtype=int)
        centers = pts.mean(axis=0, keepdims=True)
    else:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(pts)
        labels, centers = km.labels_, km.cluster_centers_
    order = np.lexsort((centers[:, 1], centers[:, 0]))
    remap = {int(old): new for new, old in enumerate(order)}

    inner = poly.buffer(-wall_clear)
    op_lines = [LineString(np.asarray(o)) for o in (openings or [])] \
        if opening_clear > 0 else []
    clusters = []
    for old in range(len(centers)):
        cid = remap[old]
        c_members = member_idx[labels == old]
        clear = [i for i in c_members
                 if not inner.is_empty and inner.contains(Point(*xz[i]))]
        relaxed = len(clear) == 0
        cand = c_members if relaxed else np.asarray(clear)
        far = [i for i in cand
               if all(ln.distance(Point(*xz[i])) >= opening_clear
                      for ln in op_lines)]
        op_relaxed = len(far) == 0 and bool(op_lines)
        if far:
            cand = np.asarray(far)
        picks = greedy_pick(xz, cand, centers[old], per_cluster)
        clusters.append({'center_id': cid,
                         'center_xz': centers[old].tolist(),
                         'picks': picks, 'clearance_relaxed': relaxed,
                         'opening_relaxed': op_relaxed})
    clusters.sort(key=lambda c: c['center_id'])
    return clusters


def find_missing_clusters(xz, outside_idx, eps=0.5, min_samples=5,
                          min_frames=30, min_hull_area=1.5):
    '''DBSCAN the outside-all-rooms frames; big dense clusters = rooms CAGE
    probably missed.

    eps=0.5 m: median frame spacing is 0.17 m, so a continuous walk chains
    into one cluster while gaps > ~1 m keep separate areas apart.
    min_samples=5 only drops isolated pose outliers. A cluster is accepted as
    a suspected missing room if it has >= min_frames frames (~5 m of walking)
    AND its convex hull covers >= min_hull_area m2 (rules out
    standing-in-place clumps).
    Returns (accepted, rejected):
      accepted [{'member_idx', 'hull_xz', 'n_frames', 'hull_area'}], sorted
      by n_frames desc; rejected is the same shape for logging.
    '''
    from shapely.geometry import MultiPoint
    from sklearn.cluster import DBSCAN

    accepted, rejected = [], []
    if len(outside_idx) < min_samples:
        return accepted, rejected
    pts = xz[outside_idx]
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(pts).labels_
    for lab in sorted(set(labels) - {-1}):
        members = outside_idx[labels == lab]
        hull = MultiPoint([tuple(p) for p in xz[members]]).convex_hull
        area = float(hull.area)
        info = {'member_idx': members,
                'hull_xz': (np.asarray(hull.exterior.coords)[:-1].tolist()
                            if hull.geom_type == 'Polygon' else
                            np.asarray(hull.coords).tolist()),
                'n_frames': int(len(members)), 'hull_area': round(area, 2)}
        if len(members) >= min_frames and area >= min_hull_area:
            accepted.append(info)
        else:
            rejected.append(info)
    accepted.sort(key=lambda c: -c['n_frames'])
    return accepted, rejected
