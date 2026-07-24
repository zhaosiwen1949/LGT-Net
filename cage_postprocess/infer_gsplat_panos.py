'''Batch LGT-Net inference over the rendered virtual panos (Method B, step 3).

Runs the standard LGT-Net inference path (VP-leveling preprocess + manhattan
post-processing) over <dataset_dir>/gsplat_render/renders/*.png and saves the
RAW intermediates needed to map predictions back into the COLMAP world:

  <out>/<name>_layout.npz
      raw_depth      [W]    model depth output (camera-height units)
      raw_xyz        [W,3]  depth2xyz(raw_depth), model frame (x right, y down,
                            z front), floor plane at y=+1
      processed_xyz  [N,3]  manhattan-regularised floor corners, same frame
      ratio          ()     ceiling height / camera height
      vp             [3,3]  vanishing directions written by preprocess
                            (pano_lsd frame; row 0 = vertical)
  <out>/<name>_vp.txt      preprocess cache (same vp)
  <out>/<name>_pred.png    visualisation (unless --no-vis)
  <out>/index.json         run manifest (cfg, ckpt dir, per-frame ratio, time)

The saved xyz stays in LGT-Net's internal normalized frame -- deliberately NOT
utils/writer.xyz2json, which bakes in a 1.6 m camera height plus an extra
R_180/x-flip output transform. cage_bev_compare.py consumes the npz directly.

Run inside the LGT-Net conda env from the repo root (imports need it):
    python cage_postprocess/infer_gsplat_panos.py --limit 2
    python cage_postprocess/infer_gsplat_panos.py --cfg src/config/mp3d.yaml
'''
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, _ROOT)

from config.defaults import get_config
from models.build import build_model
from postprocessing.post_process import post_process
from utils.conversion import depth2xyz
from utils.logger import get_logger
from utils.misc import tensor2np
from inference import preprocess, visualize_2d


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dataset_dir', default='src/datasets/huizhongbeili-106')
    ap.add_argument('--cfg', default='src/config/mp3d.yaml')
    ap.add_argument('--ckpt', default=None,
                    help='explicit checkpoint .pkl (bypasses the ckpt-dir '
                         'scan, e.g. to pick a specific zind snapshot)')
    ap.add_argument('--out', default=None,
                    help='output dir (default src/output/<scene>/'
                         'gsplat_infer_<cfg tag>)')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--post_processing', default='manhattan')
    ap.add_argument('--limit', type=int, default=0, help='only first N frames')
    ap.add_argument('--no-vis', action='store_true')
    args = ap.parse_args()
    args.mode = 'test'
    return args


def main():
    args = parse_args()
    logger = get_logger()
    config = get_config(args)
    if 'cuda' in args.device and not torch.cuda.is_available():
        args.device = 'cpu'
    config.defrost()
    config.TRAIN.DEVICE = args.device
    config.freeze()

    scene = os.path.basename(os.path.normpath(args.dataset_dir))
    tag = config.TAG
    out_dir = args.out or os.path.join('src', 'output', scene,
                                       'gsplat_infer_%s' % tag)
    os.makedirs(out_dir, exist_ok=True)

    model, _, _, _ = build_model(config, logger)
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=torch.device(args.device))
        state = ckpt['net'] if isinstance(ckpt, dict) and 'net' in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        logger.info('Overrode weights from %s' % args.ckpt)
    model.eval()

    renders = sorted(glob.glob(os.path.join(args.dataset_dir, 'gsplat_render',
                                            'renders', '*.png')))
    if args.limit:
        renders = renders[:args.limit]
    if not renders:
        raise SystemExit('no renders under %s/gsplat_render/renders'
                         % args.dataset_dir)
    print('[infer] %d rendered panos -> %s (cfg %s, tag %s)'
          % (len(renders), out_dir, args.cfg, tag))

    index = {'cfg': args.cfg, 'ckpt_override': args.ckpt, 'tag': tag,
             'dataset_dir': args.dataset_dir, 'frames': {}}
    t0 = time.time()
    with torch.no_grad():
        for i, path in enumerate(renders):
            name = os.path.splitext(os.path.basename(path))[0]
            img = np.array(Image.open(path)
                           .resize((1024, 512), Image.Resampling.BICUBIC))[..., :3]
            vp_cache = os.path.join(out_dir, '%s_vp.txt' % name)
            img, vp = preprocess(img, vp_cache_path=vp_cache)
            img = (img / 255.0).astype(np.float32)

            dt = model(torch.from_numpy(
                img.transpose(2, 0, 1)[None]).to(args.device))
            raw_depth = tensor2np(dt['depth'][0])
            ratio = float(tensor2np(dt['ratio'][0])[0])
            processed_xyz = post_process(raw_depth[None],
                                         type_name=args.post_processing)[0]
            raw_xyz = depth2xyz(raw_depth)

            np.savez(os.path.join(out_dir, '%s_layout.npz' % name),
                     raw_depth=raw_depth, raw_xyz=raw_xyz,
                     processed_xyz=processed_xyz, ratio=ratio, vp=np.asarray(vp))
            if not args.no_vis:
                dt['processed_xyz'] = processed_xyz[None]
                visualize_2d(img, dt, show=False,
                             save_path=os.path.join(out_dir,
                                                    '%s_pred.png' % name))
            index['frames'][name] = {'ratio': round(ratio, 4),
                                     'n_corners': int(len(processed_xyz))}
            print('  [%2d/%d] %s  ratio %.3f  corners %d'
                  % (i + 1, len(renders), name, ratio, len(processed_xyz)))

    index['seconds'] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, 'index.json'), 'w') as f:
        json.dump(index, f, indent=2)
    print('[infer] done in %.1fs -> %s/index.json' % (index['seconds'], out_dir))


if __name__ == '__main__':
    main()
