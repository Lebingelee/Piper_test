python script/evaluate_critics_live.py \
  -c critic_only_training/realman_itqc_reward/1:1_mc_0.8/critic_only_ckpt.pth \
   --config critic_only_training/realman_itqc_reward/1:1_mc_0.8/config_1:1.yaml \
   -d demos_RL/Stack_SF/Stack_SF.h5\
   -w 200
