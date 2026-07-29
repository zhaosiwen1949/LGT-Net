# LGT-Net

全景图室内布局估计（LGT: Layout with Geometry-aware Transformer）。原始论文代码 +
本地扩展 `cage_postprocess/`（虚拟全景 → LGT-Net → CAGE 户型 BEV 对比）。

## 环境

- `conda activate LGT-Net`（Apple Silicon；torch 2.2.2 CPU 推理，约 4–5.5 s/张）
- pylsd 已替换为 OpenCV LSD（`preprocessing/pano_lsd_align.py`，M 系芯片无 pylsd 预编译库）
- `visualization/obj3d.py` 的 open3d 为延迟导入（torch 与 open3d 各带 libomp，
  同时加载会在 macOS 段错误；不用 `--output_3d` 时不会触发）

## 模型 checkpoint

- 位置：`checkpoints/SWG_Transformer_LGT_Net/<tag>/`，`<tag>` 由 `--cfg` 的 yaml 决定
  （`src/config/mp3d.yaml` → mp3d，`zind.yaml` → zind；zind 是 depth=6，mp3d 是 depth=8，
  **配置与权重必须配套**，加载是 strict=False 配错不报错）
- 选择逻辑（`models/base_model.py load`）：目录内文件名含 `_best_` 优先（test 模式），
  其次 `_last_`；都没有则加载第一个 .pkl（mp3d 的 best.pkl 走此分支）

## 推理

```bash
python inference.py --cfg src/config/mp3d.yaml \
    --img_glob 'src/datasets/<scene>/*.jpg' --output_dir src/output/<scene> \
    --post_processing manhattan
```

- `--img_glob` 要通配符模式（目录不行）
- preprocess（VP 找平）只在 manhattan 后处理时执行，vp 缓存 `<name>_vp.txt` 存 3 行
  曼哈顿主方向；预测坐标在**找平后**的相机帧（反投影世界坐标需撤销，见下）

## cage_postprocess（虚拟全景 → 户型 BEV 对比）

从 uLayout 方法B 移植：CAGE 户型 + 3DGS 渲染 60 张理想虚拟全景（复用 uLayout 渲染图）→
LGT-Net 逐张估计布局 → 撤销 VP 找平旋转 + 已知位姿反投影世界 → 房间 union 与 CAGE
多边形算 IoU。**关键**：反投影必须 (1) 从 `*_vp.txt` 重建并撤销找平旋转，(2) 用
poses metadata 的真实相机高（1.75 m）恢复尺度（不能用 writer.py 硬编码的 1.6 m），
(3) 绕开 `xyz2json`（它带额外 R_180+x 取反输出变换）。公式与推导见
`cage_postprocess/README.md`。

```bash
python cage_postprocess/infer_gsplat_panos.py --cfg src/config/mp3d.yaml
python cage_postprocess/cage_bev_compare.py \
    --infer_dir src/output/huizhongbeili-106/gsplat_infer_mp3d
```

数据目录约定：`src/datasets/<scene>/{floorplan/, sparse/0/, point_cloud.ply, gsplat_render/}`。
huizhongbeili-106 实测 mean IoU：grid 采样 mp3d 0.684 / zind 0.706；trajectory
采样 zind 0.750（最优）/ mp3d 0.548（mp3d 在门洞附近视点会把邻房并入，轨迹采样
下必须用 zind）。uLayout 参考 ≈0.60。

视点采样有两种模式：`--sampler grid`（CAGE 多边形网格，默认）与
`--sampler trajectory`（真实相机轨迹选点，解决不可达点/CAGE 漏房/合房覆盖不足；
产物在 `gsplat_render_traj/`，两步工作流 `--plan-only` 先出预览确认；轨迹来自
`sparse/0/frames.bin`，164B/记录，见 `cage_postprocess/traj_sampling.py`）。
下游用 `--render_dir` 切换数据集（infer 的 index.json 会记录，bev_compare 自动跟随）。

`refine_cage_rooms.py` 用 LGT 逐帧 BEV 环修正 CAGE 房间错分/错合（欠分割拆、
过分割合；GT 不参与决策，splits/openings 元数据只做候选）：先 `--dry_run` 看证据表，
正式跑写 `floorplan/*_refined_polys.json`（pixel 是权威几何，不覆盖原件）+
`refined_selection.json`，然后 bev_compare 传 `--selection` 重算、CAGE 环境跑
eval_floorplan.py 验收。

**分组做在帧级**（`group_frames` 对房内帧环取连通分量），不是视点簇级——这既让只被
单帧看到的子房能被检出（r1 储物间只有 1 帧观测），也让门洞相机无需特判（门洞帧被
一侧空间主导、自然并入该组；另一个房的帧谁都不沾、自成一组）。

**面积阈值已全部换成真实 SfM 轨迹**（`sparse/0/frames.bin` 2661 帧，`--h-band
0.9 1.9` 过滤后 1982 帧，比虚拟视点密两个量级）：G3 = 各组独占区内 ≥`--g3-traj 5`
个真实位姿（原"独占区 ≥0.8㎡"删除——面积不构成房间，被人走进去才是）；
合并侧目标房 ≥`--merge-traj 5` 个位姿即否决，同时定死 splits 记录的方向
（r3 有 86 个直接出局，r4 恰好 0 个）。G3b（各组独占区内站过该组自己的相机）保留，
因为轨迹点不带分组标签，只有它能把独占区绑到声称它的那组帧上。
余量很大：G3 实测 13/67/76/257 个位姿，合并侧 0 vs 其余房 58–559。

huizhongbeili-106（zind，54 与 75 视点结论一致）：自动 split r0（新墙差 GT 9cm）、
split r1（差 15cm）、merge r4→r3 → **GT 房间匹配 9→11、F1 0.750→0.880**。
注意 GT/BEV 的 mean IoU 会因新拆出的小房拉低均值而略降（0.811→0.826、0.755→0.743），
衡量分割正确性看匹配数与 F1。剩余未匹配：储物间（CAGE 轮廓只盖到它的 54%，
需扩边界而非切边界）、p6（餐厅切出的 2.2㎡ 角，合并判据两道否决拦住）。

采样端配套（新默认）：`--min-clusters 2 --min-cluster-area 2.0` 保证 ≥2㎡ 的房
都有 ≥2 簇，`--opening-clear 0.5` 让选点避开门洞；huizhongbeili-106 由此 54→75
视点（`gsplat_render_traj2/`，已渲染推理）。
