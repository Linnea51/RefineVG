python3 inference_rsvg.py --dataset_file rsvg --num_queries 10 --with_box_refine --binary --freeze_text_encoder \
--resume rsvg_dirs/800_swin_t_p4w7/checkpoint0005.pth --backbone swin_t_p4w7 --rsvg_path data/DIOR_RSVG_addpatch

# python3 inference_rsvg.py --dataset_file rsvg_mm --num_queries 10 --with_box_refine --binary --freeze_text_encoder \
# --resume rsvg_mm_dirs/800_swin_t_p4w7/checkpoint0069.pth --backbone swin_t_p4w7
