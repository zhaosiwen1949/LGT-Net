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
huizhongbeili-106 实测 mean IoU：mp3d 0.684、zind 0.706（uLayout 参考 ≈0.60；
zind 对住宅场景溢出更少，推荐默认）。
