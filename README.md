# torch-mobilenet

MobileNetV1 (Pytorch implementation) -- see https://arxiv.org/abs/1704.04861 for the paper.

MobileNetV2 (Pytorch implementation) -- see https://arxiv.org/abs/1801.04381 for the paper.


## dataset

### coco

1. convert coco annotation file to readable.

```bash
python3 script/json_formatter.py -i data/coco/annotations/
```



## library

1. build library

```bash
cd lib/
./make.sh
```


## trainnig

1. training mobilenet on classifier dataset.
```bash
python3 train.py train -m checkpoint -w checkpoint/MobileNetV2_224_epoch:0048.pt

```

2. training mobile-faster-rcnn on coco

```bash
 CUDA_VISIBLE_DEVICES=0 python3 trainval_net.py \
                    --dataset coco --net mobilenetv1_224_100 \
                    --bs 1 --nw 4 \
                    --lr 0.001 --lr_decay_step 5 \
                    --use_tfb \
                    --cuda
```
##  testing

1. testing mobilenet on classifier dataset.

```bash
```

2. testing mobile-faster-rcnn on coco

```bash
 CUDA_VISIBLE_DEVICES=0 python3 test_net.py \
                    --dataset coco --net mobilenetv1_224_100 \
                    --load_dir models \
                    --checkepoch 1\
                    --checkpoint 234531\
                    --cuda
```
