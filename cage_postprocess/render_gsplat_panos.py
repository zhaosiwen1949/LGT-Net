'''Render equirectangular panoramas from a 3DGS .ply with gsplat360 (CUDA).

Self-contained: needs only torch / numpy / imageio (or PIL) plus an installed
gsplat360 (https://github.com/zcq15/gsplat360, the gsplat fork with the
equirectangular camera model). Copy this script together with point_cloud.ply
and gsplat_render/cameras.json (from gen_gsplat_cameras.py) to the CUDA
machine and run:

    python render_gsplat_panos.py --ply point_cloud.ply \
        --cameras cameras.json --out renders/

then copy renders/ back to <dataset_dir>/gsplat_render/renders/.

Conventions baked in (do not change one without the other):
  - cameras.json holds OpenCV camtoworld matrices; gsplat360 rasterization
    wants viewmats = world-to-camera, so inv(camtoworld) is passed.
  - gsplat360 equirect ignores Ks (identity is passed); width must be
    2 x height.
  - The camtoworld from gen_gsplat_cameras.py is now the physically-upright
    camera Ry(theta) (world up is -Y, image-down = world +Y). With that camera
    gsplat360 renders ceiling-at-top / floor-at-bottom in the real-pano
    convention directly, so NO flip is applied by default. (An earlier version
    used an inverted camera + a 180-deg image flip as a band-aid; that has been
    removed -- see --rot180 for the rare inverted case.) The pano frame ->
    world rotation is Ry(theta) (cage_bev_compare --pano-model exact).

Local dry run (no gsplat360/CUDA needed) to sanity-check the .ply parsing and
tensor shapes:
    python render_gsplat_panos.py --ply ../src/huizhongbeili-106/point_cloud.ply \
        --cameras ../src/huizhongbeili-106/gsplat_render/cameras.json --dry
'''
import argparse
import json
import os
import re

import numpy as np


def load_gaussian_ply(path):
    '''Standard 3DGS binary ply -> dict of raw numpy arrays.

    Applies the activations rasterization expects: exp(scales),
    sigmoid(opacity); SH: f_dc [N,3] + f_rest [N,45] (channel-major:
    45 = 3 channels x 15 coeffs) -> colors [N,16,3] for sh_degree=3.
    '''
    with open(path, 'rb') as f:
        header = b''
        while not header.endswith(b'end_header\n'):
            header += f.readline()
        off = f.tell()
    if b'binary_little_endian' not in header:
        raise RuntimeError('%s: only binary_little_endian ply supported' % path)
    n = int(re.search(rb'element vertex (\d+)', header).group(1))
    names = [p.decode() for p in re.findall(rb'property float (\w+)', header)]
    col = {name: i for i, name in enumerate(names)}
    need = ['x', 'y', 'z', 'opacity', 'scale_0', 'scale_1', 'scale_2',
            'rot_0', 'rot_1', 'rot_2', 'rot_3', 'f_dc_0', 'f_dc_1', 'f_dc_2']
    missing = [k for k in need if k not in col]
    if missing:
        raise RuntimeError('%s: missing properties %s' % (path, missing))
    n_rest = sum(1 for name in names if name.startswith('f_rest_'))
    raw = np.fromfile(path, np.float32, count=n * len(names),
                      offset=off).reshape(n, len(names))

    means = raw[:, [col['x'], col['y'], col['z']]]
    quats = raw[:, [col['rot_0'], col['rot_1'], col['rot_2'], col['rot_3']]]
    scales = np.exp(raw[:, [col['scale_0'], col['scale_1'], col['scale_2']]])
    opac = 1.0 / (1.0 + np.exp(-raw[:, col['opacity']]))
    sh0 = raw[:, [col['f_dc_0'], col['f_dc_1'], col['f_dc_2']]][:, None, :]
    if n_rest:
        rest = raw[:, [col['f_rest_%d' % i] for i in range(n_rest)]]
        shN = rest.reshape(n, 3, n_rest // 3).transpose(0, 2, 1)
        colors = np.concatenate([sh0, shN], axis=1)
    else:
        colors = sh0
    sh_degree = int(round(np.sqrt(colors.shape[1]))) - 1
    if (sh_degree + 1) ** 2 != colors.shape[1]:
        raise RuntimeError('SH coeff count %d is not a square' % colors.shape[1])
    return {'means': means, 'quats': quats, 'scales': scales,
            'opacities': opac, 'colors': colors, 'sh_degree': sh_degree}


def save_png(path, arr):
    try:
        import imageio.v2 as imageio
        imageio.imwrite(path, arr)
    except ImportError:
        from PIL import Image
        Image.fromarray(arr).save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--ply', required=True)
    ap.add_argument('--cameras', required=True,
                    help='cameras.json from gen_gsplat_cameras.py')
    ap.add_argument('--out', default='renders')
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--height', type=int, default=None,
                    help='override pano height (width = 2 x height)')
    ap.add_argument('--limit', type=int, default=None,
                    help='render only the first N views (debug)')
    ap.add_argument('--rot180', action='store_true',
                    help='rotate the render 180 deg (only if the camera '
                         'convention is inverted; not needed with the upright '
                         'Ry(theta) camera)')
    ap.add_argument('--dry', action='store_true',
                    help='parse inputs, print shapes and exit (no gsplat360)')
    args = ap.parse_args()

    cams = json.load(open(args.cameras))
    meta = cams['meta']
    H = args.height or int(meta['height'])
    W = 2 * H
    views = cams['views'][:args.limit] if args.limit else cams['views']
    c2w = np.array([v['camtoworld'] for v in views], dtype=np.float64)

    g = load_gaussian_ply(args.ply)
    n = len(g['means'])
    print('[render] %d gaussians, sh_degree %d; %d views at %dx%d'
          % (n, g['sh_degree'], len(views), W, H))
    print('[render] means %s quats %s scales %s opac %s colors %s'
          % (g['means'].shape, g['quats'].shape, g['scales'].shape,
               g['opacities'].shape, g['colors'].shape))
    if args.dry:
        vm = np.linalg.inv(c2w)
        print('[render] viewmats %s ok; first camtoworld:\n%s'
              % (vm.shape, np.round(c2w[0], 3)))
        print('[render] dry run done (gsplat360 not touched)')
        return

    import torch
    from gsplat360.rendering import rasterization
    dev = torch.device(args.device)
    if dev.type == 'cuda':
        assert torch.cuda.is_available(), 'CUDA not available'
    means = torch.from_numpy(g['means']).float().to(dev)
    quats = torch.from_numpy(g['quats']).float().to(dev)
    scales = torch.from_numpy(g['scales']).float().to(dev)
    opac = torch.from_numpy(g['opacities']).float().to(dev)
    colors = torch.from_numpy(g['colors']).float().to(dev)
    viewmats_all = torch.from_numpy(np.linalg.inv(c2w)).float().to(dev)
    Ks = torch.eye(3, device=dev)[None]        # ignored by equirect model

    os.makedirs(args.out, exist_ok=True)
    for i0 in range(0, len(views), args.batch):
        vm = viewmats_all[i0:i0 + args.batch]
        out = rasterization(
            means, quats, scales, opac, colors,
            viewmats=vm, Ks=Ks.expand(len(vm), 3, 3),
            width=W, height=H, sh_degree=g['sh_degree'],
            camera_model='equirectangular', render_mode='RGB')
        rgb = out[0]                            # [B, H, W, 3] in 0..1
        img = (rgb.clamp(0, 1) * 255).byte().cpu().numpy()
        for j in range(len(vm)):
            im = img[j][::-1, ::-1] if args.rot180 else img[j]
            name = views[i0 + j]['name']
            save_png(os.path.join(args.out, name + '.png'),
                     np.ascontiguousarray(im))
        print('[render] %d/%d' % (min(i0 + args.batch, len(views)),
                                  len(views)))
    print('[render] done -> %s (rot180=%s); copy back to '
          '<dataset_dir>/gsplat_render/renders/'
          % (args.out, args.rot180))


if __name__ == '__main__':
    main()
