# cage_postprocess：虚拟全景（gsplat360）→ LGT-Net 布局估计 → CAGE BEV 户型拼接对比

从 uLayout 仓库 `cage_postprocess/方法B_虚拟全景_gsplat360渲染.md` 移植的完整流程，
布局估计部分换成 **LGT-Net**。场景数据：`src/datasets/huizhongbeili-106/`
（CAGE 户型 json + COLMAP `sparse/0` + 3DGS `point_cloud.ply` + 已渲染的虚拟全景）。

## 流程总览

```
CAGE 户型 json + sparse/0 + point_cloud.ply
        │
        ▼  gen_gsplat_cameras.py（本机；三方同世界自检 + 房间采样 60 视点 + 位姿构造）
gsplat_render/{cameras.json, virtual_camera_poses.json, selection.json, viewpoints.png}
        │
        ▼  render_gsplat_panos.py（CUDA 机；gsplat360 equirect 渲染 2048×1024）
gsplat_render/renders/*.png            ←—— 本仓库直接复用 uLayout 已渲染的 60 张
        │
        ▼  infer_gsplat_panos.py（本机；LGT-Net：VP 找平 + 前向 + manhattan 后处理）
src/output/<scene>/gsplat_infer_<tag>/{*_layout.npz, *_vp.txt, *_pred.png, index.json}
        │
        ▼  cage_bev_compare.py（本机；撤销 VP 旋转 → 反投影世界 → 房间 union → 指标）
src/output/<scene>/bev_compare_gsplat_<tag>/{bev_metrics.json, bev_overlay.png}
```

常用命令（conda `LGT-Net`，仓库根目录运行；gen 在 cage_postprocess/ 下运行）：

```bash
# 0.（可选）校验相机生成逻辑与 uLayout 拷贝版一致（不写任何文件）
cd cage_postprocess && python gen_gsplat_cameras.py \
    --dataset_dir ../src/datasets/huizhongbeili-106 --verify && cd ..

# 1. 批量推理（默认 mp3d；zind 传 --cfg src/config/zind.yaml；
#    --ckpt 可直接指定某个 .pkl 绕过目录扫描）
python cage_postprocess/infer_gsplat_panos.py --cfg src/config/mp3d.yaml

# 2. BEV 拼接对比（--debug-project 2 额外输出约定校验图；--no-undo-vp 消融）
python cage_postprocess/cage_bev_compare.py \
    --infer_dir src/output/huizhongbeili-106/gsplat_infer_mp3d
```

## 关键问题：LGT-Net 前/后处理对相机位姿的影响及撤销

`virtual_camera_poses.json` 描述的是**原始渲染全景**的相机位姿，而 LGT-Net 推理前会做
VP 找平（`inference.py preprocess` → `rotatePanorama`），对全景施加一个 3D 旋转——
预测结果落在"找平后"的相机帧里，不能直接套用位姿反投影。必须重建该旋转并精确撤销。

### 各环节是否引入变换（调查结论）

| 环节 | 变换 | 处理 |
|---|---|---|
| `preprocess`（VP 找平） | 唯一的图像几何变换：`rotatePanorama(img, vp[2::-1])`，旋转 `R = inv(V.T)`，`V = vp[2::-1]`（`*_vp.txt` 的 3 行主方向逆序） | 从 vp.txt 重建并撤销（下式） |
| 模型输出 `depth2xyz` | 无旋转；模型帧 x右/y下/z前（z=图像中心列），以相机高归一化（地板 y=+1），`ratio`=天花高/相机高 | 尺度用真实相机高恢复 |
| manhattan 后处理 `fit_layout` | **无旋转**（纯 xz 轴对齐量化规整），`processed_xyz` 与 `depth2xyz` 同帧 | 无需处理 |
| 输出 json `xyz2json` | 额外 `R_180`（绕 y 转 180°）+ x 取反 + 硬编码相机高 1.6m | **完全绕开**：直接存内部 `processed_xyz`（npz） |

### 反投影公式（cage_bev_compare.py 实现）

对每帧模型输出角点 `p_model`（找平帧、归一化、地板 y=+1）：

```
V      = vp[2::-1]                    # {name}_vp.txt 的 3 行逆序（行向量）
M      = [[1,0,0],[0,0,-1],[0,1,0]]  # pano_lsd 约定(x横/y前/z上) → 模型约定(x右/y下/z前)
R_undo = M @ V.T @ M.T               # 撤销 VP 找平（V.T 是 rotatePanorama 所施旋转的精确逆）
X_w    = R_c2w @ (h_cam · R_undo @ p_model) + C

R_c2w  = poses[name].R 的转置 = Ry(theta_deg=−8.5°)   # 虚拟相机按构造朝向已知（exact 模型）
C      = poses[name].camera_center
h_cam  = metadata.floor_y − C[1] ≈ 1.75 m             # 高斯真实地面；不能用 writer.py 的 1.6
BEV 多边形 = X_w[:, [0, 2]]
```

推导要点：
- `rotatePanorama` 对场景点施加 `R = inv(V.T)`（pano_lsd 约定下 `p_aligned = R @ p_orig`），
  撤销即左乘 `V.T`——它是 `R` 的**精确**逆，vp.txt 6 位小数导致的轻微非正交不影响；
- vp 行序：`vp[0]`=竖直主方向、`vp[1]`/`vp[2]`=水平主方向
  （`pano_lsd_align.py findMainDirectionEMA` 的排序与符号规约）；
- LGT-Net 模型帧与 OpenCV 全景帧（gsplat 渲染相机）恒等，故 `pano→world = Ry(θ)` 直接套用；
- 世界上方 = −Y（重力 +Y），`h_cam = floor_y − cam_y` 为正。

### 运行时自检（脚本自动执行）

- `det(V) > 0`（det<0 说明镜像，直接报错）；
- `|vp[0]·ẑ| > 0.999`（虚拟全景近水平，VP 竖直方向应≈(0,0,±1)）；
- `R_undo` 残余旋转角统计（虚拟全景应接近恒等；>5° 打警告）；
- 天花校验：`ratio × h_cam` vs 真值 `cam_y − ceil_y = 0.610 m`（非循环的模型精度信号）；
- 每帧多边形应包含自身相机位置；
- `--debug-project N`：把 CAGE 房间多边形经**正向链**投回找平全景画墙线（红=地面高度、
  蓝=天花高度），墙线须与图像中墙面交线重合——能暴露 M/转置/乘序的任何错误，
  是约定正确性的决定性检查。

## 文件与数据格式

### 输入（`src/datasets/<scene>/`）

- `floorplan/*_aligned_polys.json` — CAGE 输出；`rooms[i].world_mm` 按米读（MVS 约定），
  经 `cage_common.load_cage` 反解（undo_yaw → 逆 up-axis 重排）到世界 (x,z)
- `sparse/0/points3D.bin`（+ `points3d_cache.npz` 缓存）— 同世界自检用
- `point_cloud.ply` — 3DGS 标准 62-float ply；自检 + 测真实地面/天花平面用
- `gsplat_render/cameras.json` — 渲染器输入（camtoworld=Ry(θ)、θ、floor_y/ceil_y、2048×1024）
- `gsplat_render/virtual_camera_poses.json` — world→camera（qvec/tvec/R/camera_center），
  metadata 含 theta_deg/floor_y/ceil_y/height_above_floor
- `gsplat_render/selection.json` — 房→视点映射（rooms[].frames[].frame = 视点名 rXX_cY_sZ）
- `gsplat_render/renders/*.png` — 渲染全景 2048×1024，60 张

### 推理产物（`gsplat_infer_<tag>/`）

- `{name}_layout.npz`：`raw_depth[256]`、`raw_xyz[256,3]`、`processed_xyz[N,3]`
  （manhattan 角点，模型帧、归一化）、`ratio`、`vp[3,3]`
- `{name}_vp.txt`（preprocess 缓存）、`{name}_pred.png`（可视化）、`index.json`（清单）

### 对比产物（`bev_compare_gsplat_<tag>/`）

- `bev_metrics.json`：逐房 iou / coverage(=|U∩C|/|C|) / overflow(=|U−C|/|U|) /
  overflow_into（溢入邻房 top3）/ overflow_outside_m2 / 面积 / frame_iou，
  以及 vp 残余角与天花误差统计（指标定义与 uLayout cage_bev_compare 一致）
- `bev_overlay.png`：CAGE 房间（填充+房号）vs 逐帧预测多边形（细线）+ 相机点，俯视

## 验证结果（huizhongbeili-106，2026-07-24）

- `gen_gsplat_cameras.py --verify` 全过：三方同世界（CAGE↔sparse 跨度比 ≤1.0%、
  ply↔sparse 盒内 96.4%），theta/camera_y/floor_y/ceil_y 与 uLayout 版一致
  （|err|≤4e-5），60 视点采样坐标逐点吻合（偏差 0.0000 m）
- `--debug-project`：CAGE 墙线投影与找平全景中的墙面交线目视重合（朝向/高度/手性全对）
- VP 残余旋转：mean 0.63°、max 10.49°（max 出现在 1㎡ 退化小房 r04）；
  `--no-undo-vp` 消融 mean IoU 不变（0.684）——虚拟全景近水平，撤销链无害且正确
- 天花校验：`ratio×1.75` 对真值 0.610 m 平均偏 +0.07 m（mp3d）/ −0.02 m（zind）
- **BEV 指标**：mp3d mean IoU **0.684**、zind mean IoU **0.706**（uLayout 参考 ≈0.60）；逐房：

  | room | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | mean |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | LGT-Net(mp3d) | 0.82 | 0.48 | 0.71 | 0.47 | 0.07 | 0.82 | 0.88 | 0.92 | 0.78 | 0.69 | 0.80 | 0.76 | 0.684 |
  | LGT-Net(zind) | 0.83 | 0.46 | 0.66 | 0.60 | 0.34 | 0.90 | 0.76 | 0.86 | 0.75 | 0.68 | 0.90 | 0.74 | 0.706 |
  | uLayout 参考 | 0.815 | 0.444 | 0.693 | 0.288 | 0.048 | 0.721 | 0.797 | 0.523 | 0.748 | 0.586 | 0.814 | 0.697 | 0.598 |

  zind（住宅数据训练）整体更稳：溢出显著更少（overflow 普遍 <0.15），大房 r5/r10 达 0.90，
  连 1㎡ 退化小房 r4 也从 0.07 提到 0.34。r1 两个模型都低——预测面积只有 CAGE 的一半，
  属场景/CAGE 分割问题。r4（1㎡ 储物间）视点贴墙，本质是采样问题而非变换问题。

## 与 uLayout 版的差异

- 布局来源：LGT-Net 输出**墙体布局多边形**（manhattan 角点，闭合多边形直接可用）；
  uLayout 输出逐列边界曲线，需按 `cs = h/tan(v)` 重建**自由空间 ring**。两者在开门/遮挡处
  行为不同（LGT-Net 倾向补全墙体，ring 会顺开口溢出），逐房 IoU 不严格可比但同量级。
- LGT-Net 需要 VP 找平前处理，多了"从 vp.txt 撤销旋转"一步（uLayout 的 exact 模型直接用
  `Ry(θ)`）；本仓库照常保留找平（模型训练分布如此），在反投影时精确撤销。
- 尺度：LGT-Net 内部以相机高归一化，用 poses metadata 的真实相机高（1.75 m）恢复米制；
  绝不能走 `utils/writer.xyz2json`（硬编码 1.6 m 且带额外输出帧变换）。
