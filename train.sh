export CUDA_VISIBLE_DEVICES=1
nohup python2 trainval_net.py  \
     --dataset pascal_voc --net res101   --bs 1   --lr 0.001 --lr_decay_step 5   \
     --cuda \
     > 3Gru_3.log 2>&1 &
