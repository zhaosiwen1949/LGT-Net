  conda activate LGT-Net
  KMP_DUPLICATE_LIB_OK=TRUE python inference.py \
    --cfg src/config/mp3d.yaml \
    --img_glob 'src/datasets/xinghewan/*.jpg' \
    --output_dir src/output/xinghewan \
    --post_processing manhattan