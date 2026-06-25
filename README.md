# 📈 Predict 1-Year US Stock Returns from Fundamentals

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/predict-1-year-us-stock-returns-from-fundamentals)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Tabular-📊-orange?style=flat-square)]()

本项目专为 Kaggle 竞赛 **"Predict 1-Year US Stock Returns from Fundamentals"** 打造。任务的核心是利用上市公司的底层量化财务基本面数据，精准预测其未来一年的股票收益率。

---

## 🏆 竞赛提交结果

| 评估维度 | 结果详情 |
| :--- | :--- |
| **Kaggle 账户** | Laiyupeng |
| **最终排名 (Rank)** | 🥇 **Top 83** |
| **平台最终得分** | `14765.52270` |
| **最终生成文件** | `submission.csv` |

> ⚠️ **数据说明**：`train.csv`、`test.csv` 和 `sample_submission.csv` 均归属于 Kaggle 官方赛事数据集。根据开源规范，本仓库**不上传**任何原始数据文件。

---

## 🛠️ 方案与核心方法概述

本次方案聚焦于**表格机器学习（Tabular ML）**。整体管道（Pipeline）涵盖：数据流读取、深度缺失值与异构值分析、高度定制化财务特征构造、严谨的时间切分验证、多模型异构训练、多阶集成融合以及最终的线性校准。

### 1. 📅 验证策略：时间切分验证（Time-based Split）
拒绝使用会导致时序信息泄露的随机交叉验证（Random CV）。
* **训练集**：2019 年 - 2021 年数据
* **验证集**：2022 年数据
* **优势**：这种切分方式完美模拟了测试集来自“未来年份”的真实外推场景，能大幅提升线下验证（Local OOF）与线上榜单（LB）的分数一致性。

### 2. 📊 特征工程（Feature Engineering）
针对财务报表高度稀疏、量纲差异巨大的特点，构建了以下六大特征版块：

* **数据质量与统计特征**：各样本当前的缺失值数量、缺失比例、零值数量、负值数量。
* **经典财务比率**：如净利润 / 营收（Profit Margin）、股东权益 / 总资产（ROE 基础分量）、长期债务 / 总资产等。
* **估值衍生特征**：市销率（P/S）、市净率（P/B）、市盈率的倒数（E/P，即盈利收益率）及其对应的正负状态标记。
* **近似市值代理（Market Cap Proxy）**：利用 `price_to_sales * revenue_ttm` 隐式还原公司市值。
* **行业内相对特征（Sector-Relative Metrics）**：计算各财务指标相对所属行业（Sector）中位数的偏离度。
* **匿名 Ticker 纵向特征**：同一匿名 Ticker 内部的均值、标准差、绝对排名（Rank）以及历史偏离均值度。

### 3. 🤖 异构模型库（Model Zoo）
基模型全面覆盖了主流的强力表格机器学习算法：
* 🌲 **梯度提升树族**：`XGBoost` / `LightGBM` / `CatBoost` / `HistGradientBoosting`
* 🪵 **经典集成树族**：`RandomForest` / `ExtraTrees`

### 4. 🔗 阶梯式模型融合（Ensemble Strategy）
方案拒绝了单一模型的产出，采用**三阶段融合管道**：
1. **贪心融合（Greedy Blending）**：初步筛选表现优异的模型组合。
2. **非负权重优化（Non-Negative Bound Optimization）**：通过求解约束优化问题，自动分配各基模型的最佳非负权重，防止过拟合。
3. **线性校准（Linear Calibration）**：对融合后的预测结果进行最后一步全局线性平移与缩放，使其分布更逼近真实收益率。

---

## 🚀 运行指南

### 1. 环境准备
确保你的本地环境已安装 Python 3.8+ 及相关主流数据科学库：
```bash
pip install xgboost lightgbm catboost scikit-learn pandas numpy
