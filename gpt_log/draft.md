松灵机械臂主机 /home/lebinge/Desktop/wry 

rsync -avh --progress --partial --append-verify /home/lebinge/Desktop/wry wry@172.16.67.110:/home/wry/Desktop/ssh


/home/lebinge/Desktop/wry
154309

rsync -avh --progress --partial --append-verify /home/lebinge/Desktop/Piper_test/data/merged_cpiql_dac_hitl_deploy_runner zyf@172.16.67.43:/home/zyf/ssh
/home/lebinge/Desktop/wry


rsync -avh --progress --partial --append-verify /home/zyf/lly_project/Piper_test/run_results/piper_dual_merged_cpiql_dac_finetune/model_config.yaml lebinge@172.16.67.48:/home/lebinge/Desktop/


rsync -avh --progress --partial --append-verify /home/zyf/lly_project/Piper_test/data/ lly24229070@114.214.255.70:/data2/group_朱聪聪/lly24229070/lly/program_VLA/Agent_lab/data


接着我在观察V(k=0)就V(k=1)曲线时，发现它们居然完全重合了，说明模型根本没有去学习那些失败的数据。我通过调试训练代码观察dataset中的数据结果，发现了一个很严重的事实：dataset中的progress_mask会将成功轨迹设置为1，失败轨迹设置为0这导致了在V的损失中_progress_anchor_loss，模型直接忽略了失败轨迹的retuen信息，


codex已经把我们的itqc训练完成并且实现评估，评估对象选择了data/test中四个未见过的数据集，traj_0和traj_29表示成功，其余表示失败，我经过训练之后发现它仅仅只在训练集