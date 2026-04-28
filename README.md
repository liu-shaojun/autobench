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
  host_model_dir: /home/intel/weights   # 改成你机器上模型的路径
```

### server_defaults（服务启动参数）

所有模型共用的 vLLM 启动参数，一般不用动。

### tests（测试矩阵）

```yaml
tests:
  gsm8k:
    enabled: true          # 关掉就不跑准确性
    num_questions: 100
  perf:
    enabled: true
    concurrency: [1, 4, 6, 8]
    input_len: [1024, 8192, 16384, 32768, 65536]
    output_len: [1024]
```

### models（模型列表）

每个模型只写跟默认不同的字段。同一模型不同量化会自动用 `模型名_量化` 区分：

```yaml
models:
  - name: Qwen3.5-27B
    ze_affinity_mask: "0,1,2,3"
    server:
      args:
        tensor-parallel-size: 2
        max-model-len: 35000
        max-num-batched-tokens: 8192
        quantization: sym_int4

  - name: Qwen3.5-27B           # 同名模型，不同量化
    ze_affinity_mask: "0,1,2,3"
    server:
      args:
        tensor-parallel-size: 2
        max-model-len: 35000
        max-num-batched-tokens: 8192
        quantization: fp8
```

单个模型可覆盖测试矩阵：

```yaml
    tests:
      perf:
        input_len: [1024, 2048]   # 这个模型只跑这两个
```

## 实时看进度

终端里会显示 rich 表格，也可以另开窗口看日志：

```bash
# 所有模型的合并日志（推荐）
tail -f results/$(ls -t results | head -1)/logs/all.log

# 某个模型的日志
tail -f results/$(ls -t results | head -1)/logs/Qwen3.5-27B_sym_int4.log

# 实时状态 JSON
watch -n 2 cat results/$(ls -t results | head -1)/status.json
```

## 看结果

跑完后结果在 `results/<时间戳>/` 下：

```
results/20260427_154503/
├── summary.csv        # 所有模型所有组合的汇总表
├── status.json        # 最终状态
└── logs/
    ├── all.log        # 合并日志
    ├── Qwen3.5-27B_sym_int4.log
    └── ...
```

**summary.csv** 是最终结果，每行一个测试组合，包含：

| 列 | 含义 |
|---|---|
| model | 模型标签（名称_量化） |
| gsm8k_accuracy | 准确率 |
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

## 其他

- `./run.sh --dry-run` — 只打印命令不执行，用来检查配置
- `Ctrl+C` — 随时中断，会自动清理 container 并保存已有结果
- 每个模型跑 perf 前会自动 warmup 一次，warmup 数据不计入结果
