项目简介

本项目用于 Kaggle 竞赛 'Predict 1-Year US Stock Returns from Fundamentals'。任务是根据上市公司的基本面数据预测未来一年股票收益率，并生成符合竞赛格式的 'submission.csv'。

竞赛链接：

https://www.kaggle.com/competitions/predict-1-year-us-stock-returns-from-fundamentals

提交结果

Kaggle 账号：Laiyupeng

最终排名：83

平台分数：14765.52270

最终提交文件：'submission.csv'

说明：'train.csv'、'test.csv' 和 'sample_submission.csv' 来自 Kaggle 竞赛页面，仓库中不上传原始数据文件。

方法概述

本次方案主要使用表格机器学习方法。整体流程包括数据读取、缺失值分析、财务特征构造、时间切分验证、多模型训练、模型融合和提交文件生成。

验证方式采用时间切分：使用 2019-2021 年数据训练，使用 2022 年数据验证。这样比随机切分更接近测试集来自未来年份的场景。

特征工程包括：

- 缺失数量、缺失比例、零值数量、负值数量
- 财务比率，例如净利润 / 营收、股东权益 / 总资产、长期债务 / 总资产
- 估值相关特征，例如市销率、市净率、市盈率的倒数和正负标记
- 近似市值特征，例如 'price_to_sales * revenue_ttm'
- 行业内相对特征，例如相对 sector 中位数的偏离
- 同一匿名 ticker 内部的均值、标准差、排名和偏离均值

使用的模型包括：

- XGBoost
- LightGBM
- CatBoost
- ExtraTrees
- RandomForest
- HistGradientBoosting

最终没有直接选择单个模型，而是先做贪心融合，再使用非负权重优化，并进行一次线性校准。

运行方式

将 Kaggle 下载的数据文件放在同一目录下：

'''text
train.csv
test.csv
sample_submission.csv
'''

然后运行：

'''bash
python kaggle_stock_solution_no_comments.py
'''

运行结束后会生成 'submission.csv'。
