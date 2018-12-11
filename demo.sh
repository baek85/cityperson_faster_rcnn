#nohup python3 -m visdom.server &
CUDA_VISIBLE_DEVICES=1 python3 train.py train --lr=1e-3 --env='pedestrian-1e-3' --plot-every=50 --caffe-pretrain
#CUDA_VISIBLE_DEVICES=0 python3 train.py train --lr=1e-4 --env='pedestrian-1e-4' --plot-every=50 --caffe-pretrain
#CUDA_VISIBLE_DEVICES=1 python3 train.py train --data voc --env='voc-caffe' --plot-every=100 --caffe-pretrain

