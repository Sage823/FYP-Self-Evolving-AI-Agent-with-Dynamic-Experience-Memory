# 任务清单

> 来源: 用户要求与现有 FYP proposal
> 生成时间: 2026-10-26
> 说明: 项目缺少 design.md，本次按用户授权直接进入实施，只处理可运行性与验证中发现的明确问题。

## 实施任务

- [x] 【配置隔离环境】创建项目本地 Python 虚拟环境并安装开发依赖
  - 目标: 让项目依赖不再依赖电脑上的偶然全局状态
  - 涉及文件: `.venv/`（Git 忽略）
  - 预期结果: Python、pytest、scipy、pandas、scikit-learn 与向量存储依赖均可导入
- [x] 【运行测试与校验】执行单元测试、benchmark 校验和离线端到端流程
  - 目标: 验证当前代码是否真正能运行，并暴露代码级问题
  - 涉及文件: `tests/`、`src/experience_agent/`、临时运行目录
  - 预期结果: 测试与离线流程通过，或得到可复现的失败证据
- [x] 【修复明确问题】仅修复验证中确认的逻辑、配置或可复现性缺陷
  - 目标: 消除阻碍正常运行或造成错误结论的问题，不扩大研究范围
  - 涉及文件: 以测试结果为准
  - 预期结果: 修复有回归测试保护，全部检查重新通过
- [x] 【执行一次实验】在不泄露凭据的前提下运行一次完整实验
  - 目标: 优先完成离线全流程；若本机已有 API 凭据，再尝试真实模型流程
  - 涉及文件: `results/`（Git 忽略）
  - 预期结果: 产生一次可解释的运行结果，并明确区分离线验证与真实模型证据
- [x] 【校正安全指标标签】统一准确率下降的计算方向、字段名与报告文案
  - 目标: 防止把 `N - L` 的数值误写成 `L - N`
  - 涉及文件: `src/experience_agent/analysis.py`、`tests/test_analysis.py`
  - 预期结果: 指标含义明确，并由回归测试锁定
- [x] 【缩小迁移实验范围】只对达到反馈错误阈值的题族执行后续迁移实验
  - 目标: 避免对不会进入主要分析的题族产生无效 API 调用
  - 涉及文件: `src/experience_agent/experiment.py`、`tests/test_experiment.py`
  - 预期结果: 调度范围与分析范围一致，并由回归测试锁定
- [x] 【接入持久化 ChromaDB】用 Chroma 集合替换 FAISS 内存索引，并保留本地哈希 embedding
  - 目标: R/L 经验分别持久化到实验目录，关闭后可重新打开和检索
  - 涉及文件: `src/experience_agent/memory.py`、`src/experience_agent/experiment.py`、依赖声明与测试
  - 预期结果: ChromaDB 不下载默认 embedding 模型，持久化与检索均由测试锁定
- [x] 【接入 DeepSeek 本地配置】通过受 Git 忽略的 `.env` 加载官方端点、模型和密钥
  - 目标: 密钥不进入版本控制，命令行可直接运行真实实验
  - 涉及文件: `.env`、`.env.example`、`config/experiment_v0.json`、配置与客户端代码
  - 预期结果: 使用 DeepSeek 官方兼容接口完成一次真实运行，日志不泄露密钥
- [x] 【完整验证与项目结构交付】安装新依赖并执行测试、离线实验和真实实验
  - 目标: 确认迁移无回归，并提供可读的项目结构和运行结果
  - 涉及文件: `.venv/`、`results/runs/`
  - 预期结果: 全部测试通过，至少一次真实运行完成或给出可复现的外部错误
- [x] 【生成 BBH 风格 v1 benchmark】用确定性生成器构造四类多步推理题及 oracle 答案
  - 目标: 形成 8 道反馈题、16 道迁移题和 8 道无关题，不复制公开 BBH 原题
  - 涉及文件: `src/experience_agent/bbh_benchmark.py`、`data/benchmark_bbh_v1.jsonl`、测试
  - 预期结果: 固定种子重复生成相同数据，现有 benchmark 校验器通过
- [x] 【真实 probing 与难度校准】用 DeepSeek 运行反馈题，确认至少一个题族产生真实错误
  - 目标: 只根据模型实际错误决定哪些题族进入纠错流程
  - 涉及文件: `config/experiment_bbh_v1.json`、`results/runs/`
  - 预期结果: 若首版仍全对，则提高客观任务复杂度并重新探测
- [x] 【完成真实纠错闭环】执行 N/R/L 条件并验证 Chroma 写入、检索和统计输出
  - 目标: 得到至少一个由真实错误触发的经验记忆与后续迁移结果
  - 涉及文件: `results/runs/`、Chroma 集合、分析报告
  - 预期结果: 真实运行状态为 completed，报告清楚区分工程证据与研究结论

## 完成状态

> 进度: 12/12 已完成

## 2026-10-26 follow-up

- [x] 区分完整、错误、截断、空回复、格式错误与接口错误
- [x] 生成独立 baseline、feedback、validation、follow-up 数据分区
- [x] 使用独立 validation 数据校准检索阈值和 top-k
- [x] 增加 hashing 与 sentence-transformers 可比 embedding 接口
- [x] 比较两个 backend，并记录校准配置
- [x] 增加面向初学者的操作、验证和 Git 推送手册
- [x] 冻结 v3 开发集并将每族 follow-up 扩展到 20 道
- [x] 使用扩大样本重跑真实 DeepSeek 实验
- [x] 记录扩样结果、截断排除和解释边界
