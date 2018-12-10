#python3 train.py train --env='fasterrcnn-caffe' --plot-every=100 --caffe-pretrain
CUDA_VISIBLE_DEVICES=1 python3 train.py train --data voc --env='voc-caffe' --plot-every=100 --caffe-pretrain

