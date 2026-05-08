# autobench

vLLM 多模型自动化性能测试框架。编辑一个配置文件，一键跑完所有模型的准确性 + 性能测试。

## 快速开始

```bash
# 1. 编辑配置
vim configs/config.yaml

# 2. 跑
./run.sh
```

## 配置说明

所有配置在 `configs/config.yaml` 一个文件里，分四段：

### docker（机器相关）

换机器主要改 `host_model_dir`（宿主机上模型目录）：

```yaml
docker:
  image: amr-registry.caas.intel.com/intelanalytics/llm-scaler-vllm:v0.14.0-b8.3-0428
  host_model_dir: /home/intel/weights   # 改成你机器上模型的路径
```

### server_defaults（服务启动参数）

所有模型共用的 vLLM 启动参数，一般不用动。

### tests（测试项开关 + 默认参数）

```yaml
tests:
  smoke:
    enabled: true              # 5 个快速 prompt 验证服务正常
  gsm8k:
    enabled: true
    num_questions: 100
    script: default            # "default" 或 "no_think"
  lm_eval:
    enabled: true
    timeout_sec: 3600
    tasks:
      - name: arc_challenge
        limit: 500
      - name: mmlu_high_school_computer_science
        limit: 50
      - name: mmlu_college_computer_science
        limit: 50
      - name: truthfulqa_mc2
        limit: 100
  perf:
    enabled: true
    concurrency: [1, 4]
    input_len: [1024, 8192, 16384, 32768, 65536]
    output_len: [2048]
```

#### gsm8k script 选项

- `default` — 下载远程 gsm8k_eval.py（标准 completion 模式）
- `no_think` — 用本地 gsm8k_no_think.py（chat 模式 + `enable_thinking: False`，适合 Qwen3.x 27B sym_int4 等需要禁用 thinking 的模型）

### models（模型列表）

每个模型只写跟默认不同的字段。自动用 `模型名_tp数_量化` 区分（如 `Qwen3.6-27B_tp2_sym_int4`）：

```yaml
models:
  - name: Qwen3.6-27B
    ze_affinity_mask: "4,5,6,7"
    server:
      args:
        tensor-parallel-size: 2
        max-model-len: 70000
        max-num-batched-tokens: 8192
        quantization: sym_int4
    tests:
      gsm8k:
        script: no_think       # 这个模型用 no_think 脚本

  - name: Qwen3.6-27B           # 同模型不同 TP
    ze_affinity_mask: "4,5,6,7"
    server:
      args:
        tensor-parallel-size: 4
        max-model-len: 70000
        max-num-batched-tokens: 8192
        quantization: sym_int4
```

单个模型可覆盖任意测试参数：

```yaml
    tests:
      gsm8k:
        enabled: false         # 不跑 gsm8k
      lm_eval:
        enabled: false         # 不跑 lm_eval
      perf:
        input_len: [1024, 2048]   # 只跑这两个输入长度
```

## 测试流程

每个模型按顺序执行：

1. **Smoke** — 5 个快速 prompt，验证服务基本可用
2. **GSM8K** — 数学准确性测试（100 题）
3. **LM-Eval** — ARC/MMLU/TruthfulQA 等评估套件
4. **Warmup** — 一次小请求预热（结果丢弃）
5. **Perf** — 性能矩阵（concurrency × input_len × output_len）

任何一步失败都只记录不中断，继续下一步/下一个模型。

## 实时看进度

终端里会显示 rich 表格（Model / TP / Stage / Smoke / GSM8K / LM-Eval / Perf / Error）。

另开窗口看日志：

```bash
# 所有模型的合并日志（推荐）
tail -f results/$(ls -t results | head -1)/logs/all.log

# 某个模型的日志
tail -f results/$(ls -t results | head -1)/logs/Qwen3.6-27B_tp2_sym_int4.log

# 实时状态 JSON
watch -n 2 cat results/$(ls -t results | head -1)/status.json
```

## 看结果

跑完后结果在 `results/<时间戳>/` 下：

```
results/20260429_055252/
├── summary.csv        # 所有模型所有组合的汇总表
├── status.json        # 最终状态
└── logs/
    ├── all.log        # 合并日志
    ├── Qwen3.6-27B_tp2_sym_int4.log
    └── ...
```

**summary.csv** 每行一个测试组合：

| 列 | 含义 |
|---|---|
| model | 模型标签（名称_tp_量化） |
| tp | tensor parallel size |
| smoke_ok | smoke 测试结果 |
| gsm8k_accuracy | GSM8K 准确率 |
| arc_challenge 等 | lm-eval 各 task 的 acc |
| concurrency | 并发数 |
| input_len / output_len | 输入/输出长度 |
| successful_requests | 成功请求数 |
| request_throughput | 请求吞吐 (req/s) |
| output_throughput | 输出 token 吞吐 (tok/s) |
| total_throughput | 总 token 吞吐 (tok/s) |
| ttft_mean_ms | 首 token 延迟 (ms) |
| tpot_mean_ms | 每 token 延迟 (ms) |
| itl_mean_ms | token 间延迟 (ms) |

用 Excel 或 `column -s, -t summary.csv` 查看。

如果中途 Ctrl+C 导致 summary.csv 数据不全，可以从 log 重新生成：

```bash
python3 -m autobench.parse_logs results/<run_id>
```

## 独立脚本

不需要整个框架，只想手动跑单个模型时：

```bash
# Smoke test（在 vLLM server 已起来的机器上）
bash smoke_test.sh Qwen3.6-27B 9005

# 性能 benchmark（在 container 内）
bash bench_perf.sh Qwen3.6-27B /llm/models/Qwen3.6-27B 9005
```

## 其他

- `./run.sh --dry-run` — 只打印命令不执行，用来检查配置
- `./run.sh --no-ui` — 不显示 rich 表格，纯日志输出
- `Ctrl+C` — 随时中断，会自动清理 container 并保存已有结果
- 每次启动自动清理上次残留的 `autobench-*` container
- 每个模型跑 perf 前会自动 warmup 一次，warmup 数据不计入结果
