#!/usr/bin/env bash
set -x

srun --partition=VA --job-name=derived --gres=gpu:4 --ntasks=1 --ntasks-per-node=8 -x BJ-IDC1-10-10-16-[44,48,56-58] --kill-on-bad-exit=1 \
python MGPU_search_arch.py \
--gpu_ids 4,5,6,7 \
--gen_bs 120 \
--dis_bs 120 \
--dataset cifar10 \
--bottom_width 4 \
--img_size 32 \
--max_epoch_G 5 \
--arch search_both_cifar10 \
--latent_dim 120 \
--gf_dim 160 \
--df_dim 80 \
--g_spectral_norm False \
--d_spectral_norm True \
--g_lr 0.0002 \
--d_lr 0.0002 \
--beta1 0.0 \
--beta2 0.9 \
--init_type xavier_uniform \
--n_critic 5 \
--val_freq 10 \
--derive_freq 1 \
--derive_per_epoch 16 \
--draw_arch False \
--exp_name search/bs120-dim160 \
--num_workers 40 \
--gumbel_softmax True