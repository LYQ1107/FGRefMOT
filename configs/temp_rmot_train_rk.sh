EXP_DIR=default_debug_rk
OUT='/data1/liuyeqiang/TempRMOT/default_debug_rk/Results/'
TRAIN_LOG_FILE="$OUT/${EXP_DIR}/train_log.txt"
PID_FILE="$OUT/${EXP_DIR}/train_pid.txt"

mkdir -p "$OUT/${EXP_DIR}"

python3  -m torch.distributed.launch --nproc_per_node=2 --master_port 29504 \
   --use_env main.py \
   --meta_arch temp_rmot \
   --use_checkpoint \
   --dataset_file e2e_rmot \
   --epoch 5 \
   --with_box_refine \
   --lr_drop 40 \
   --lr 1e-4 \
   --lr_backbone 1e-5 \
   --pretrained /data1/liuyeqiang/TempMam/exps/official_checkpoint/checkpoint_0060.pth\
   --output_dir exps/${EXP_DIR} \
   --save_dir exps/${EXP_DIR} \
   --batch_size 1 \
   --sample_mode random_interval \
   --sample_interval 1 \
   --sampler_steps 60 80 90 \
   --sampler_lengths 4 4 4 4 \
   --update_query_pos \
   --merger_dropout 0 \
   --dropout 0 \
   --random_drop 0.1 \
   --fp_ratio 0.3 \
   --query_interaction_layer QIM \
   --data_txt_path_train /data1/liuyeqiang/RMOT/datasets/data_path/refer-kitti.train \
   --rmot_path /data1/liuyeqiang/RMOT/data/Dataset/refer-kitti \
   --hist_len 4 \
   --refer_loss_coef 2 \




