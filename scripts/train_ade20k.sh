python train.py --name oasis_ade20k --dataset_mode ade20k --gpu_ids 0,1,2,3 \
--dataroot path_to_folder/ADEChallengeData2016 --batch_size 32 --freq_print 10000 \
--checkpoints_dir path_to_a_folder_where_experiments_should be_saved \
--lr_g 0.0001 --lr_d 0.0004 --lambda_labelmix 5
