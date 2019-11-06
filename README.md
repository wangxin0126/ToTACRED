# ToTACRED

本项目的目的是为了将SemEval2010数据集转换成Tacred数据集格式

## 目录结构

```
.
├── RAEDME.md
├── SemEval2010_task8_all_data
├── ToTacredResult
│   ├── test.json
│   └── train.json
├── log
│   ├── error.txt
│   └── more_than_two.txt
├── toTACRED.py
```

- toTACRED.py读取SemEval2010_task8_all_data并转换成Tacred数据集格式
- ToTacredResult放置运行结果
- log记录一些运行程序时的错误信息

## 程序运行

- 本程序主要利用stanfordNLP工具来对数据集进行分析处理，主要用到
  - stanford-postagger-2018-10-16
  - stanford-ner-2018-10-16
  - stanford-parser-full-2018-10-17
- 这些stanfordNLP包可以在网上自行下载，下载好之后，更改程序toTACRED.py中的配置即可运行程序

## 注意

- 生成的数据集效果可能并不是十分理想，如果有问题可以检查代码，或者根据Tacred数据集的含义自己撸一份代码