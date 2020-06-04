# ChineseTextClassification
自然语言处理之中文文本分类（以垃圾短信识别为例）

## 数据集 ##
- 格式：标签\t文本
- 标签：正样本标签为1，表示垃圾短信；负样本标签为0，表示正常短信
- 文本：短信文本

## 环境依赖 ##
- Python3.6
- jieba
- Scikit-learn

## 分类算法 ##
- SVM：支持向量机
> 可根据需要替换为其他分类模型


## 使用说明 ##
```
python train.py
```

## 流程图 ##
![流程图](https://github.com/junxincai/ChineseTextClassification/blob/master/flow%20chart.jpg)
