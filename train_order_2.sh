CUDA_VISIBLE_DEVICES=0 python main.py --essay_prompt_id_train 8 --essay_prompt_id_test 8 --target_model cent_hds_order --init_lr 0.001 > log/cent_hds_order_change_lr/p8_lr1e-3.log
CUDA_VISIBLE_DEVICES=0 python main.py --essay_prompt_id_train 1 --essay_prompt_id_test 1 --target_model cent_hds_order --init_lr 0.001 > log/cent_hds_order_change_lr/p1_lr1e-3.log
CUDA_VISIBLE_DEVICES=0 python main.py --essay_prompt_id_train 2 --essay_prompt_id_test 2 --target_model cent_hds_order --init_lr 0.001 > log/cent_hds_order_change_lr/p2_lr1e-3.log
