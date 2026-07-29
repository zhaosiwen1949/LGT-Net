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
#    --ckpt 可直接指定某个 .pkl 绕过目录扫描；
#    轨迹采样集加 --render_dir src/datasets/<scene>/gsplat_render_traj）
python cage_postprocess/infer_gsplat_panos.py --cfg src/config/mp3d.yaml

# 2. BEV 拼接对比（--debug-project 2 额外输出约定校验图；--no-undo-vp 消融；
#    render_dir 默认自动从 infer 的 index.json 读取，无需重复传）
python cage_postprocess/cage_bev_compare.py \
    --infer_dir src/output/huizhongbeili-106/gsplat_infer_mp3d
```

## 视点采样：grid 与 trajectory 两种模式

`gen_gsplat_cameras.py --sampler {grid,trajectory}`（默认 grid）。

**grid（原始）**：CAGE 多边形内缩 → 按 ~9㎡ 切子区 → 中心+抖动各 3 点。问题：
采样点可能不可达（床/家具上）；CAGE 漏房无法覆盖；CAGE 把多个实际房间合成
一个时中心点覆盖不足。

**trajectory（改进）**：从 `sparse/0/frames.bin`（COLMAP 3.11+ rig 格式，164B/记录，
`rig_from_world` 即 pano_camera0 外参，`C=-Rᵀt`）读取全部真实相机轨迹，多层次聚类选点：

1. **按 CAGE 房间分类**（shapely point-in-polygon）+ 高度带过滤（离地 0.9–1.9m）
2. **房内 KMeans**：簇数 = ⌈房间面积/9㎡⌉（`random_state=0`，簇 id 按簇心坐标排序，
   全链路确定性）；簇内优先离墙 ≥0.3m 的点（无达标点则放宽，记 `clearance_relaxed`），
   每簇选 min(3, n) 个真实点：质心最近 1 + 贪心最远点采样 2
3. **漏房检测**：房外点（剔除高度带外与距房边界 ≤0.15m 的墙厚/穿门噪声）DBSCAN
   （eps=0.5m，min_samples=5）；簇 ≥30 帧且凸包 ≥1.5㎡ 判为"疑似 CAGE 漏房"，
   也出视点（命名 `u%02d_c%d_s%d`）、渲染、推理、画 overlay，但无 CAGE 真值不算 IoU
4. **无覆盖房间**：轨迹从未进入的 CAGE 房标 `no_coverage`，不出兜底点（暴露
   CAGE 幻觉房/采集盲区本身；`--grid-fallback` 可强制网格兜底）

真实轨迹只提供 (x,z) 可达位置；渲染仍统一高度 1.75m + 朝向 Ry(θ)，exact 模型与
反投影链完全不变。产物写 `gsplat_render_traj/`（不碰旧 `gsplat_render/` 与其渲染图）。

**两步确认工作流**：

```bash
cd cage_postprocess
# 第一步：只出预览（viewpoints_preview.png + selection_draft.json），人工确认选点
python gen_gsplat_cameras.py --dataset_dir ../src/datasets/huizhongbeili-106 \
    --sampler trajectory --plan-only
# 第二步：确认后写正式四件套（会校验与 draft 的视点集合一致，防中途改参）
python gen_gsplat_cameras.py --dataset_dir ../src/datasets/huizhongbeili-106 \
    --sampler trajectory
# 然后拷 point_cloud.ply + gsplat_render_traj/cameras.json + render_gsplat_panos.py
# 到 CUDA 机渲染，渲完拷回 gsplat_render_traj/renders/，再跑 infer + bev_compare
```

huizhongbeili-106 实测（2026-07-24）：2661 帧轨迹，54 视点（12 房 19 簇）；
2660/2661 帧落在 CAGE 多边形内 → 此场景 CAGE 无漏房；r4（1㎡ 储物间）轨迹从未
进入 → no_coverage（正是 grid 模式下 IoU 仅 0.07/0.34 的房间——网格点落在不可达
位置）。frames.bin 解析与 uLayout camera_poses.json 逐帧对拍 diff=0。

### trajectory 采样 BEV 结果（54 视点渲染后，2026-07-25）

VP 残余旋转 max 从 grid 的 10.49° 降到 **1.70°**（不可达退化视点消失）。
逐房 IoU（排除 no_coverage 的 r4，与 grid 同口径 11 房均值对比）：

| room | 0 | 1 | 2 | 3 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | mean(11房) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| grid + zind | 0.83 | 0.46 | 0.66 | 0.60 | 0.90 | 0.76 | 0.86 | 0.75 | 0.68 | 0.90 | 0.74 | 0.740 |
| **traj + zind** | **0.85** | **0.57** | 0.65 | 0.60 | **0.92** | **0.79** | **0.87** | 0.66 | 0.63 | **0.93** | **0.79** | **0.750** |
| grid + mp3d | 0.82 | 0.48 | 0.71 | 0.47 | 0.82 | 0.88 | 0.92 | 0.78 | 0.69 | 0.80 | 0.76 | 0.739 |
| traj + mp3d | 0.36 | 0.69 | 0.60 | 0.16 | 0.82 | 0.14 | 0.60 | 0.79 | 0.46 | 0.78 | 0.63 | 0.548 |

- **zind + trajectory 是最优组合（0.750）**：大房 r10 0.90→0.93、r5 0.90→0.92、
  过道房 r1 0.46→0.57、r11 0.74→0.79，天花误差 mean −0.009m。
- **mp3d 在轨迹采样下明显退化（0.548）**：真实轨迹点多在行走路径/门洞附近，
  mp3d（Matterport 大宅训练）倾向把门洞外可见空间并入本房（r0 溢入 r10 达 28㎡、
  r6 溢入 r5 12㎡），zind（住宅训练）对开口收敛得多。**轨迹采样下模型选择更关键，
  用 zind。**

## 房间分割修正：refine_cage_rooms.py（LGT 证据修 CAGE 的错分/错合）

CAGE 的房间分割有两类错误（对照 GT `CAGE/data/floorplan/<scene>_floorplan/
rooms_centerline.json`；注意 `*_eval.json` 是评测报告不是 GT）：
**欠分割**（多间真实房并成一间，如 r0 = 卧室B+过道B 的 L 形）与**过分割**
（一间拆成多间，如 r3/r4 = 厨房被 `align_info.splits` 程序劈开）。
`refine_cage_rooms.py` 用 LGT-Net 逐帧 BEV 环作主判据修正（GT 不参与决策，
CAGE 元数据 splits/openings 只做候选生成与佐证），并用 `sparse/0` 的**真实 SfM
相机轨迹**（2661 帧，远密于虚拟视点）作"这块地方有没有人走进去"的物理证据：

```bash
# 先 dry_run 看证据表（帧级分组、spill、各门限逐项数值），确认后去掉 --dry_run
python cage_postprocess/refine_cage_rooms.py \
    --infer_dir src/output/huizhongbeili-106/gsplat_infer_zind_traj2 --dry_run
# 正式跑：写 floorplan/<scene>_..._refined_polys.json（不覆盖原件）
#        + refine_report.json / refined_selection.json / refine_overlay.png
python cage_postprocess/refine_cage_rooms.py \
    --infer_dir src/output/huizhongbeili-106/gsplat_infer_zind_traj2
# 用 refined 房间重算 BEV（复用已有推理 npz，不重新渲染）
python cage_postprocess/cage_bev_compare.py \
    --infer_dir src/output/huizhongbeili-106/gsplat_infer_zind_traj2 \
    --selection src/output/huizhongbeili-106/refine_zind_traj2/refined_selection.json \
    --out src/output/huizhongbeili-106/bev_compare_gsplat_zind_traj2_refined
# GT 验收（CAGE 环境）
cd /Users/ke/Workspace/CAGE && conda run -n CAGE python eval_floorplan.py \
    --pred .../floorplan/<scene>_..._refined_polys.json \
    --gt_dir data/floorplan/<scene>_floorplan --output_dir infer_out/<scene>
```

### 视点采样的两项配套改进（2026-07-28）

拆分判定要求房内 ≥2 个视点簇，但原簇数公式 `k=⌈面积/9㎡⌉` 让所有小于 9㎡ 的房
（r1/r2/r3/r6/r7/r9/r11）都只有 1 簇，天然无法检测——r1 就是这样漏掉的。
另外站在门洞里的相机会预测出跨两个空间的布局，污染簇一致性。两项对策：

- `--min-clusters 2 --min-cluster-area 2.0`（新默认）：面积 ≥2㎡ 的房至少 2 簇。
  huizhongbeili-106：18 簇/54 视点 → **25 簇/75 视点**，大房 r10 仍是 4 簇不碎片化。
- `--opening-clear 0.5`（新默认）：选点优先离 CAGE `openings` 门洞线段 ≥0.5m，
  簇内无达标点时放宽（记 `opening_relaxed`）。实测离门洞 <0.5m 的视点从
  **10/54 降到 1/75**。

新采样集在 `gsplat_render_traj2/`（旧的 `gsplat_render_traj/` 未动），已渲染并推理
（`gsplat_infer_zind_traj2/`）。75 视点对原始 CAGE 的 BEV 基线 0.755（vs 54 视点
0.750），增益集中在小房与被开口截断的房：r3 +0.029、r5 +0.028、r7 +0.025。

### 分组粒度：帧级连通分量（门洞相机因此不需要单独处理）

拆分判定的分组做在**帧级**而不是视点簇级（`group_frames()`）：房内两帧的环
`ov=inter/min` ≥`--t-link 0.30` 就连边，取连通分量。这一条决定了两件事：

- **只被单帧看到的子房也能被检出**。簇级分组下，KMeans 按相机位置分簇，
  少数派帧会被塞进某个簇里再被簇内投票压掉——r1 的储物间只有 1 帧观测到
  （轨迹在那半边只有 12/96 帧），簇级永远拆不出来。
- **门洞相机不再需要专门识别**。门洞帧的预测被其中一侧的空间主导，于是自然并入
  那一侧的分量；真正在另一个房里的帧则谁都不沾、自成一组。实测 r0：门洞帧
  `r00_c2_s1` 与 6 个卧室帧重叠 0.66–0.93、与 2 个走廊帧全为 0.00；r1：
  `r01_c0_s1` 度数为 0。两者都自动落到正确的组，无需任何特判。

（此前用"与同簇其他帧的最大 IoU"当跨空间判据剔除门洞帧，已删除：该判据无法区分
"站在门洞里"和"人在另一个被合并的房里"——两者都表现为与兄弟帧零重叠——
因而恰好把 r1 唯一的储物间证据也删掉了。`cage_bev_compare --skip-cross-space`
一并移除。）

### 判据（全部在 CAGE ps 帧做，曼哈顿方向=坐标轴）

- **拆分**（≥2 帧的房才检测）：帧级连通分量得到 ≥2 组后，两组需过全部保守门限：
  G1 互斥 `ov<0.15`、G2 组内容于父房 ≥40%、**G3 各组独占区内有 ≥`--g3-traj 5`
  个真实 SfM 轨迹位姿**、**G3b 各组独占区内至少站过该组自己的一台相机**
  （`--cam-tol 0.25` 容差）、G4 沿单轴投影可分（重叠 ≤20%）、
  G5 切完两侧一致性 ≥75%。
  G3 原来是"独占区 ≥0.8㎡"，已换成轨迹判据：**面积不构成房间，被人走进去才是**。
  轨迹用 `sparse/0/frames.bin`（2661 帧，`--h-band 0.9 1.9` 过滤后 1982 帧），
  比 75 个虚拟视点密两个量级；实测每个真实房 13–28 位姿/㎡，
  r1 拆出的 0.96㎡ 储物间有 13 个（原面积门槛下它是 0.96 vs 0.8 险过），
  预测碎片则是 0。G3b 补 G3 补不了的一件事：轨迹点不带分组标签，
  G3b 才能把独占区与"声称它的那组帧"绑定。
  新墙位置 = 两组环的接缝扫描 argmin（1px 步进，平台取中点），
  距既有墙线 <0.4m 才吸附（r0 实际 -4.21，不吸附凹角 -5.39）。
- **合并**（结构先验 AND 环证据，纯环溢出永不单独触发）：候选来自
  `align_info.splits` 记录或"微房(<1.5㎡ 或短边<0.7m) 且轨迹位姿 <`--merge-traj 5`"；
  需邻房环 union 溢入目标 ≥50% 且次名 <一半。两道否决：
  **目标内有 ≥5 个真实轨迹位姿**（人在里面走过 = 真房），或目标自覆盖 ≥50%。
  轨迹否决同时定死了 splits 记录的方向——r3/r4 互为候选，r3 有 86 个位姿直接出局，
  r4 恰好 **0** 个（这也是 CAGE 把厨房劈出一条 0.98㎡ 窄条的直接证据）。
  r8→r0 的 3.6㎡ 门洞溢出连候选都不是。
- 编号：先合并释放槽位再拆分填入（r4 槽位复用），槽位不够则追加（r1 拆出的
  r12）；未动房 id 全稳定、pixel 逐字节不变；openings 按几何重派；
  refined json 里 **pixel 是权威几何**（eval_floorplan.py 只读
  pixel+normalization），world_mm 由 pixel 反算。

### 结果（huizhongbeili-106，zind，2026-07-28）

自动检出并执行三个操作：`split r0`（卧室B / 过道B，新墙 ps x=-4.207，GT 墙 -4.3，
差 **9cm**）、`split r1`（过道A / 储物间，新墙 ps y=3.216，GT 墙 3.063，差 **15cm**）、
`merge r4→r3`（厨房）。其余 9 个房都是单连通分量，不误触发。
**54 视点集与 75 视点集给出完全相同的三个操作**，结论对视点密度不敏感；
把 G3 面积门槛换成轨迹判据后，三个操作与输出 pixel 几何**逐字节不变**
（面积门槛本就不是当前场景的瓶颈，换掉是为了去掉那个拍脑袋的数）。

判据留出的余量（越大越不怕换场景）：G3 独占区轨迹 13 / 67 / 76 / 257 个位姿
（门限 5）；合并侧 r4 = 0 个位姿，其余 11 个房 58–559 个（门限 5）——
0 与 58 之间没有中间地带，这条判据在本场景几乎不可能擦边翻转。

| 指标 | 修正前 | 簇级分组 | **帧级分组（当前）** |
|---|---|---|---|
| GT 房间匹配 | 9/12 | 10/12 | **11/12** |
| GT F1 | 0.750 | 0.833 | **0.880**（P 0.846 / R 0.917）|
| GT mean IoU（匹配房） | 0.811 | 0.837 | 0.826 |
| BEV mean IoU | 0.755（11/12 房） | 0.759（12/12） | 0.743（13/13） |

逐房 GT IoU：卧室B 0.805→**0.942**、厨房 0.656→**0.780**、过道B 未匹配→**0.840**、
过道A 未匹配→**0.682**；其余房间一位不变。

两个指标反向要看明白：**GT mean IoU 和 BEV mean IoU 都是"已匹配/已覆盖房间的均值"，
新拆出的小房分数天然低于大房，会稀释均值**。真正衡量分割正确性的是匹配数与 F1
（9→11 / 0.750→0.880）。BEV 下降还有第二层原因：它拿预测环与 **CAGE 多边形**比，
而 r1 那块 CAGE 轮廓本身就只盖到 GT 储物间的 54%，把一个错的多边形切成两个错的
多边形并不会让它和预测更吻合。

### 剩余未匹配（本方法修不了的部分）

- **GT 储物间**：拆出的 p1 有 82% 落在储物间内，但只盖住储物间面积的 47%——
  CAGE 的轮廓从一开始就只覆盖了它的 54%，缺的那块在 CAGE 里根本没有几何，
  房间**再分割**无法凭空造出来（需要的是扩边界，不是切边界）。
- **pred p6**（从餐厅切出的 2.2㎡ 角）：属过分割，但既无 `align_info.splits`
  记录、自身又有轨迹覆盖和帧证据，合并判据的两道否决都拦住了。放宽会误合真房。

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
