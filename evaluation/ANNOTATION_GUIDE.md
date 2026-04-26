# 50 条 LLM 抽取人工标注规范 (Rubric)

打开 `extraction_sample.csv`,逐行阅读 `content` (评论原文) + `extracted_*` 三列 (LLM 抽取结果),按下面的规则填 6 列 `human_*`.

## 列说明

### 1. human_aspects_correct (Y / P / N)
LLM 抽出的 aspects 准确性
- **Y** 全对: 抽出的所有 aspect 都对应原文,没有捏造或冗余
- **P** 部分对: 至少 1 个对,但有错的或多余的(评论根本没提那个点)
- **N** 全错: 基本没对应

### 2. human_aspects_missed (Y / N)
是否漏抽关键 aspect
- **Y** 漏抽: 评论里**明显**强调的 aspect 没抽到 (例:评论用一半篇幅吐槽"价格贵",aspects 里却没有)
- **N** 不漏: 核心点都抽到了

### 3. human_features_correct (Y / P / N)
features 抽取的准确性,标准同 aspects_correct

### 4. human_sentiment_correct (Y / N)
- **Y**: positive/negative/neutral 与你读完整体感受一致
- **N**: 明显错(评论全是吐槽,标了 positive;或夸奖却标 negative)

### 5. human_error_type (boundary / synonym / domain / fabrication / none)
按"主要错误"挑一个
- **boundary**: 实体边界错 (例:"电池续航差"被抽成"电池")
- **synonym**: 同义词没合并 (例:同一条评论里"快递慢"和"物流差"被抽成两个 aspect)
- **domain**: 领域错配 (例:手机壳评论里抽出"护肤效果")
- **fabrication**: 捏造 (评论根本没提的内容硬抽出来)
- **none**: 没明显错 (即 aspects=Y 且 sentiment=Y 时填 none)

### 6. human_notes (自由文本,可空)
关键案例/特别有趣的错误简短记一句,后面挑典型错误进论文用. 一般 60% 行可以留空.

## 速度参考

每条评论 60-90 秒: 读 content (~30s) + 比对抽取 (~30s) + 打分 (~15s).
50 条总计 50-75 分钟.

## 标完后

保存 CSV (Excel 默认 UTF-8 with BOM,直接保存即可),回到项目根目录运行:

```bash
python scripts/analyze_annotation.py
```

会输出表 3 数字 + 错误类型 Top 3 + 典型错误案例摘要,直接拷进论文 §4.5.
