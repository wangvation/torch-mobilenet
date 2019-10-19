# torch-mobilenet
MobileNet (Pytorch implementation) -- see https://arxiv.org/abs/1905.02244 for the paper.


## dataset

### coco

```bash
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python3 setup.py
```


## trainnig

```bash
python3 train.py train -m checkpoint -w checkpoint/MobileNetV2_224_epoch:0048.pt

```