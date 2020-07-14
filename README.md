# NMT

python+pytorch
基于LSTM实现简单的NMT

#### Dataset
本次使用用了两个数据集，一个是中文-英文数据集，一个是英语-西班牙语数据集

#### 运行
中文-英文数据集运行，以英文翻译到中文为例（按默认参数，具体设置详见代码）：
``` text
创建vocab：python vocab.py --en_cn --source_language="en"
训练模型：python test.py --train --en_cn --source_language="en"
进行翻译：python test.py --translate --en_cn
```
英语-西班牙语数据集运行，以西班牙语翻译到英语为例（按默认参数，具体设置详见代码）：
```
创建vocab：python vocab.py ----en_es
训练模型：python test.py --train --en_es
进行翻译：python test.py --translate --en_es
```

#### 结果
output中有测试集上的翻译结果，英文翻译为中文的结果较差，可能是数据集过小的原因，西班牙语翻译为英语的结果相对较好。
