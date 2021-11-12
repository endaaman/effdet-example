# effdet-example

|Original|Ground truth|Predicted(default weight)|
|---|---|---|
|![](https://github.com/endaaman/effdet-example/raw/master/example.png) | ![](https://github.com/endaaman/effdet-example/raw/master/example_gt.png)| ![](https://github.com/endaaman/effdet-example/raw/master/example_pred_default.png) |


## train

```
$ python train.py
```

## predict

```
$ python predict.py -s foo.png -c checkpoints/d0/20.pt
```
