# DIOR-RSVG dataset
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -m torch.distributed.launch --nproc_per_node=8  --master_port 29502 --use_env \
main.py --dataset_file rsvg --binary --with_box_refine \
--batch_size 4 --num_frames 1 --epochs 70 --lr_drop 40 --num_queries 10 \
--output_dir rsvg_dirs/800_swin_t_p4w7 --backbone swin_t_p4w7 --rsvg_path data/DIOR_RSVG_addpatch \
--backbone_pretrained /data/swin_tiny_patch4_window7_224.pth

# OPT-RSVG dataset
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -m torch.distributed.launch --nproc_per_node=8  --master_port 29502 --use_env \
# main.py --dataset_file rsvg --binary --with_box_refine \
# --batch_size 4 --num_frames 1 --epochs 70 --lr_drop 40 --num_queries 10 \
# --output_dir optrsvg_dirs/800_swin_t_p4w7 --backbone swin_t_p4w7 --rsvg_path data/OPT_RSVG_addpatch \
# --backbone_pretrained /data/swin_tiny_patch4_window7_224.pth

# # RSVG-HR dataset
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -m torch.distributed.launch --nproc_per_node=8  --master_port 29500 --use_env \
# main.py --dataset_file rsvg_mm --binary --with_box_refine \
# --batch_size 2 --num_frames 1 --epochs 70 --lr_drop 40 --num_queries 10 \
# --output_dir rsvg_mm_dirs/800_swin_t_p4w7 --backbone resnet50 \
# --backbone_pretrained /data/swin_tiny_patch4_window7_224.pth

