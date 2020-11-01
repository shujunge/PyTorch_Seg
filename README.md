Segmentation for everything by pytorch
=========

-[ ] 不同loss性能对比实现
-[ ] 不同验证指标添加
-[ ] 不同训练技巧
-[ ] 不同数据增强实验

## 运行命令
```
python main.py -h

python -B main.py --image_size 352 --backbone resnest101 --head PSPNet --GPUs 1,2 --batch_size 28 --aux True
```

## dataset shape:
```
image:(3,H,W)   mask:(H,W), np.unique(mask)=(0,C)
```

## model shape:
```
input:(batch_size, 3, H, W)
output:(batch_size, C, H, W)

```