# Preprocessing module

## Normalizer
Normalize origin corpus into the several sentence-by-sentence documents delimited by an empty line. Each line is a sentence.

**NOTE**: Besides the sentence punctuation (e.g. [。！？] and so on), we also consider the order number like [1. （1）1、] and so on.

```
>>> ./normalize.sh
```
More details:
```
>>> python normalize.py --help
```

## Processor

### Progress
* [2022.01.05] MLM processor, supporting WWM

### Command
```
>>> ./preprocess.sh
```
More details:
```
>>> python preprocess.py --help
```

## Index builder
### Command
```
>>> ./buildIndex.sh
```

## Pipeline
We also provide a pipeline script to finish above three procedures one by one:
```
>>> ./pipeline.sh
```
