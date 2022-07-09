# 预训练数据预处理

## Requirements

```
transformers==4.17.0
tokenizers~=0.12.1
torch
zhon
thulac
```

## 第零步：训练词表

已经训练完成，并放在`tokenizer`目录下。

## 第一步：文本正则化

文本正则化，包括断句处理，并最终以“文档”为单位划分训练集和验证集，文档和文档之间用空行间隔。

* input：语料文件所在的目录，对应`pipeline.sh`中的`RAW_CORPORA_PATH`
* output：正则化后的文档，训练集和验证集分别输出到`pipeline.sh`中的`NORMALIZED_TRAIN`和`NORMALIZED_DEV`目录中。

预计骤消耗时间10小时。

## 第二步：预处理

上一步处理的结果仍然是可读的文字格式，而输入模型的是离散的id，因此本步骤将会用tokenizer对文本做编码。处理结束后，每个sample在文件中存为一行。

* input：上一步的`NORMALIZED_TRAIN`和`NORMALIZED_DEV`。
* output：处理完的结果会写入到`PROCESSED_TRAIN`和`PROCESSED_DEV`中。

第二步总体消耗时间视分配的进程数量而定，30进程（默认）约为10~15小时。

## All-in-One

直接执行当前目录下的`pipeline.sh`即可。

注意，默认目录组织结构为：

```
${DATA_PATH}
├──15.medical_plm_corpus    # 原始语料
├──normalized               # 正则化后的train
├──normalized_dev           # 正则化后的dev
└──processed                # 预处理后的结果
   ├──train                 # 训练数据
   │  ├──epoch-0.pretrain.data.shuf
   │  ├──epoch-1.pretrain.data.shuf
   │  ├──...
   │  └──epoch-n.pretrain.data.shuf
   └──dev                   # 验证数据
      └── dev.data.all
```

可适当调整目录结构，总之最终的训练数据和验证数据存在在`${DATA_PATH}/processed`目录下。
