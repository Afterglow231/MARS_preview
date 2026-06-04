# 用 conda 配置 `mars`（按你的 CUDA 版本自动选 cu128）

这个仓库默认用 `uv`，但你也可以用 conda 创建一个可复现的
`mars` 环境来构建本地 vLLM CUDA 扩展，并运行 MARS clean 版实验。
顶层实验入口请优先看仓库根目录的 `README.md`；本文件只说明
`vllm-continuum/` 这棵源码树的 conda 构建方式。

## 1) 创建环境并安装依赖（推荐：源码编译，最稳）

在有 GPU 的机器上执行（需要联网下载 conda/pip 包）：

```bash
MODE=source bash scripts/create_continuum_vllm_conda_env.sh
conda activate mars
```

脚本会自动读取 `nvidia-smi` 的 `CUDA Version`（或 `nvcc`）来选择 PyTorch wheels（本仓库当前依赖 `torch==2.8.0`，对应 `cu128`），并用 conda 安装 CUDA toolkit（含 `nvcc`）后在本机编译 vLLM 的 CUDA 扩展（需要一些时间）。

如果自动检测失败，可手动指定：

```bash
MODE=source CUDA_VERSION_OVERRIDE=12.7 bash scripts/create_continuum_vllm_conda_env.sh
```

## 2) 启动 vLLM 服务（Continuum 模式）

```bash
conda activate mars
MODEL=${HOME}/models/Qwen3-Coder-30B-A3B-Instruct-FP8 TP=1 bash scripts/start_vllm_service.sh
```

注意：`TP` 必须 ≤ 当前可见 GPU 数；脚本会自动检查并在不匹配时直接报错。`Llama-3.1-70B` 通常需要多卡（或量化/更小模型），单卡可能会 OOM。

可选：启用 LMCache 的 CPU KV offload（示例 200GB）：

```bash
conda activate mars
LMCACHE_MAX_LOCAL_CPU_SIZE=200 \
MODEL=meta-llama/Llama-3.1-70B-Instruct TP=1 \
bash scripts/start_vllm_service.sh
```

默认会把 Continuum 的 `scheduler_timestamps` 写到 `RUN_OUTPUT_DIR`（默认 `./continuum_exp/`），用于后续分析。

## 3) 回到仓库根目录跑 MARS 实验

clean 版不包含 `mini-swe-agent/` 和本地分析脚本；环境创建脚本会在
该目录不存在时跳过 SWE-bench helper 安装。构建完成后，回到仓库根目录：

```bash
cd ..
DRY_RUN=1 bash experiments/scripts/run_policy_smoke_gpu1.sh
```

如果只想启动 standalone vLLM 服务，仍可使用：

```bash
cd vllm-continuum
MODEL=${HOME}/models/Qwen3-Coder-30B-A3B-Instruct TP=1 bash scripts/start_vllm_service.sh
```

## 4) 如果你必须本地编译（需要 nvcc）

当你有系统 CUDA Toolkit（`nvcc`）或希望自己编译时（脚本默认会用 conda 安装 `nvcc`，一般不需要你额外装）：

```bash
MODE=source bash scripts/create_continuum_vllm_conda_env.sh
```

若遇到 `nvcc`/`CUDA_HOME` 问题，请先确认：

```bash
which nvcc
echo $CUDA_HOME
```

## Troubleshooting

- 如果你看到 `terminate called after throwing an instance of 'std::bad_alloc'` 且堆栈发生在导入 `vllm._C`/启动 `vllm serve`：通常是因为启用了 `VLLM_USE_PRECOMPILED=1`（预编译 `.so` 与当前 `torch`/CUDA 组合 ABI 不匹配）。解决：用 `MODE=source` 重新安装并本机编译扩展。
- 如果你看到 `ValueError: Unsupported FA version: None`：通常说明当前
  `vllm-continuum` 源码树没有生成
  `vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so` 和
  `_vllm_fa3_C.abi3.so`，或者 Python import 到了没有本地编译产物的
  checkout。解决：在 `mars` 环境里重新执行
  `python -m pip install -e . --no-build-isolation --no-deps`，或者重新运行
  `MODE=source bash scripts/create_continuum_vllm_conda_env.sh`。不要把软链
  或复制别的机器/别的 checkout 的 `.so` 当作发布安装方案。
- 如果你看到类似 `/etc/conda/deactivate.d/deactivate-gxx_linux-64.sh: line 39: CONDA_BACKUP_CXX: unbound variable`：这是因为在 `bash` 里启用了 `set -u`（nounset），而 conda 的 activate/deactivate 脚本会引用可能未设置的 `CONDA_BACKUP_*` 变量。解决：确保 `conda activate/deactivate` 执行时关闭 nounset（脚本已做兼容处理），或手动 `set +u` 后再执行 conda 命令。
- 如果你看到 `<frozen graalpy.pip_hook>` 或 `platform.python_implementation()` 显示 `GraalVM`，并伴随 `ImportError: Failed to load PyTorch C extensions`：说明 conda 把环境解成了 GraalPy（非 CPython），PyTorch wheels 无法工作。解决：强制切回 CPython（或直接删环境重建）：
  - `conda install -n mars -c conda-forge "python=3.10.*=*cpython*"`（然后重跑脚本或重新 `pip install` torch 相关依赖）
- 如果你看到 `error: could not create 'vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so': No such file or directory`：说明源码树里缺少 `vllm/vllm_flash_attn/` 目录（或被误创建成同名文件），导致 editable 安装拷贝编译产物失败。解决：
  - `rm -f vllm/vllm_flash_attn && mkdir -p vllm/vllm_flash_attn`，然后重新运行 `python -m pip install -e . --no-build-isolation`
