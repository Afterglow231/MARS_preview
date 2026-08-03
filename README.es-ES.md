

# MARS: Co-Planificación Eficiente y Adaptativa para Sistemas Agénticos Heterogéneos

**MARS** es un prototipo de sistema de servicio para cargas de trabajo agénticas heterogéneas. Extiende vLLM con políticas de planificación para cargas de trabajo que intercalan la inferencia en GPU con la ejecución de herramientas en el lado del CPU, y proporciona una ruta de integración basada en OpenHands para la evaluación de agentes de extremo a extremo.

Este repositorio es una versión de vista previa para el [artículo de MARS](https://arxiv.org/abs/2604.26963). Incluye el código necesario para ejecutar los experimentos, el código fuente de vLLM modificado por MARS, el ejecutor de cargas de trabajo OpenHands y una carga de trabajo híbrida compacta.

Políticas de planificación admitidas:

- `mars`: co-planificación adaptativa para cargas de trabajo agénticas heterogéneas.
- `fcfs`: planificación de solicitudes en orden de llegada.
- `continuum` y `continuum_dy`: líneas base construidas a partir de la implementación oficial de [vLLM-Continuum](https://github.com/Hanchenli/vllm-continuum); `continuum_dy` lo extiende con valores de TTL computados dinámicamente.
- `autellix`: una línea base de Autellix implementada con la política de planificación PLAS propuesta en el [artículo de Autellix](https://arxiv.org/abs/2502.13965).
- `infercept`: una línea base de InferCept implementada siguiendo el principio de mínimo desperdicio del [artículo de InferCept](https://arxiv.org/abs/2402.01869) en las interceptaciones de llamadas a herramientas.

---

## Cómo Funciona

### vLLM Modificado por MARS

`vllm-continuum/` contiene el código fuente local de vLLM utilizado en los experimentos. Agrega las implementaciones del planificador y los ganchos de tiempo de ejecución necesarios para comparar MARS con las líneas base incluidas.

### Cargas de Trabajo Agénticas

Esta versión de vista previa incluye una carga de trabajo de prueba compacta en `benchmark/hybrid_workloads/test_workload.jsonl`. Este archivo es la carga de trabajo predeterminada que consumen los scripts de lanzamiento de vLLM replay y OpenHands.

### Rutas de Evaluación

El repositorio proporciona dos rutas de ejecución:

- **vLLM replay**: ejecuta la carga de trabajo a través de un motor vLLM intraproceso con ejecución simulada del lado de las herramientas.
- **Integración con OpenHands**: inicia un servidor vLLM local compatible con OpenAI y ejecuta trabajadores de OpenHands contra él para la ejecución de agentes de extremo a extremo.

---

## Estructura del Proyecto

```text
MARS/
|-- vllm-continuum/                  # MARS-modified vLLM source tree
|-- experiments/
|   |-- replay/vllm_tool_baseline/   # vLLM replay runtime
|   |-- openhands/                   # OpenHands integration and tests
|   `-- scripts/                     # Maintained launch scripts
|-- benchmark/
|   |-- hybrid_workloads/test_workload.jsonl # Test workload
|   |-- GitTaskBench/                # Referenced task assets
|   `-- terminal-bench/              # Referenced task assets
`-- LICENSE.txt
```

---

## Instalación

**Requisitos:** `conda`, `git`, `cmake`, `ninja`, `python>=3.12`, `torch==2.8.0+cu128`; GPU NVIDIA de nivel de centro de datos. Probado en H100 y H200.

```bash
# Clone the repository and enter the repo root.
git clone <repo-url> MARS
cd MARS

# Build the MARS-modified vLLM environment.
cd vllm-continuum
MODE=source bash scripts/create_continuum_vllm_conda_env.sh
conda activate mars
cd ..

# If CUDA auto-detection fails, rebuild with explicit CUDA settings:
# cd vllm-continuum
# MODE=source CUDA_VERSION_OVERRIDE=12.8 TORCH_CUDA_TAG=cu128 \
#   bash scripts/create_continuum_vllm_conda_env.sh
# conda activate mars
# cd ..

# Verify that Python imports vLLM from this repository and sees the local
# FlashAttention extension modules.
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/vllm-continuum" \
python - <<'PY'
import importlib
import vllm

print("vllm:", vllm.__file__)
for name in ("vllm.vllm_flash_attn._vllm_fa2_C", "vllm.vllm_flash_attn._vllm_fa3_C"):
    mod = importlib.import_module(name)
    print(name, mod.__file__)
PY

# Create the OpenHands worker environment.
# The vLLM service runs in conda env `mars`; OpenHands workers use this venv.
# OpenHands 1.14 requires Python >=3.12. These commands use python3.12 because
# python3.13 is not installed by default on many servers; override as needed.
OPENHANDS_VENV="${OPENHANDS_VENV:-$HOME/.venvs/openhands312}"
OPENHANDS_VENV_PYTHON="${OPENHANDS_VENV_PYTHON:-python3.12}"
"$OPENHANDS_VENV_PYTHON" -m venv "$OPENHANDS_VENV"
"$OPENHANDS_VENV/bin/python" -m pip install --upgrade pip
"$OPENHANDS_VENV/bin/python" -m pip install \
  openhands-sdk==1.14.0 \
  openhands-tools==1.14.0 \
  litellm==1.82.3
export OPENHANDS_PYTHON="$OPENHANDS_VENV/bin/python"

# Verify the OpenHands runtime.
OPENHANDS_SUPPRESS_BANNER=1 "$OPENHANDS_PYTHON" - <<'PY'
from openhands.sdk import Agent, Conversation, LLM, Tool
from openhands.tools.terminal import TerminalTool
from openhands.tools.file_editor import FileEditorTool
print("OpenHands runtime OK")
PY
```

---

## Uso

Establece `MODEL_PATH` en un directorio de modelo local antes de ejecutar experimentos en GPU. Los scripts utilizan por defecto `$HOME/models/Qwen3-Coder-30B-A3B-Instruct`.

### 1. Experimento de Reproducción con vLLM

Ejecuta la carga de trabajo de prueba a través de la línea base de vLLM replay con la política de planificación seleccionada:

```bash
conda activate mars
cd /path/to/MARS
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=$HOME/models/Qwen3-Coder-30B-A3B-Instruct \
TASKS_PATH=$PWD/benchmark/hybrid_workloads/test_workload.jsonl \
NUM_REQUESTS=100 RPS=0.2 SCHEDULING_POLICY=mars \
bash experiments/scripts/run_baseline_continuum.sh
```

### 2. Experimento de Integración con OpenHands

Inicia un servidor vLLM local compatible con OpenAI y ejecuta el ejecutor de cargas de trabajo OpenHands:

```bash
conda activate mars
cd /path/to/MARS
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=$HOME/models/Qwen3-Coder-30B-A3B-Instruct \
TASKS_PATH=$PWD/benchmark/hybrid_workloads/test_workload.jsonl \
MAX_REQUESTS=20 RPS=0.2 \
bash experiments/scripts/run_openhands_qwen3_local.sh
```


### 3. Prueba de Humo de Políticas de Planificación

Ejecuta todas las políticas de planificación en una GPU. Por defecto, la prueba de humo utiliza las primeras 200 solicitudes para el objetivo de vLLM replay y las primeras 50 solicitudes para el objetivo de OpenHands.

```bash
conda activate mars
cd /path/to/MARS
bash experiments/scripts/run_policy_smoke_gpu1.sh
```


---

## Recorrido de Extremo a Extremo

Los siguientes comandos ejecutan una canalización de verificación compacta sobre la carga de trabajo incluida:

```bash
conda activate mars
cd /path/to/MARS
export MODEL_PATH=$HOME/models/Qwen3-Coder-30B-A3B-Instruct

# Step 1: run a small vLLM replay job
CUDA_VISIBLE_DEVICES=0 NUM_REQUESTS=100 RPS=0.2 SCHEDULING_POLICY=mars \
  bash experiments/scripts/run_baseline_continuum.sh

# Step 2: run a small OpenHands job
CUDA_VISIBLE_DEVICES=0 MAX_REQUESTS=20 RPS=0.2 SCHEDULING_POLICY=mars \
  bash experiments/scripts/run_openhands_qwen3_local.sh

# Step 3: compare selected policies on GPU 0
GPU=0 POLICIES="fcfs mars" TARGETS="vllm openhands" \
  VLLM_REQUESTS=200 OPENHANDS_REQUESTS=50 \
  bash experiments/scripts/run_policy_smoke_gpu1.sh
```

---

## Notas sobre Reproducibilidad

- Todos los scripts de lanzamiento son relativos a la raíz del repositorio.
- Sobrescribe `MODEL_PATH`, `TASKS_PATH`, `PYTHON_BIN`, `MARS_PYTHON`, `OPENHANDS_PYTHON` y las variables de salida según sea necesario para tu máquina.
- Los valores predeterminados de vinculación de CPU son `CPU_GPU=0-1`, `CPU_TOOL=16-23` y `CPU_CORES=0-1,16-23`; sobrescríbelos si tu nodo tiene una distribución de CPU diferente.

---

## Solución de Problemas

`ValueError: Unsupported FA version: None`

: El árbol local de `vllm-continuum` no se construyó correctamente, o Python está importando un árbol de código fuente sin `_vllm_fa2_C.abi3.so` y `_vllm_fa3_C.abi3.so`. Reconstruye `vllm-continuum` en el entorno `mars` activo y vuelve a ejecutar la verificación de instalación.

`OPENHANDS_PYTHON not found`

: Crea el entorno de OpenHands anterior o pasa `OPENHANDS_PYTHON=/path/to/openhands/python`.

`AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended`

: El entorno `mars` resolvió un stack de tokenizador incompatible. Reinstala las versiones probadas con `python -m pip install transformers==4.55.2 tokenizers==0.21.1 huggingface-hub==0.36.2 lmcache==0.4.4`.

`Model path not found`

: Descarga o monta el modelo localmente y establece `MODEL_PATH=/path/to/model`.

---

## Próximamente

- **Reproducción completa de experimentos** — scripts de extremo a extremo para reproducir los resultados completos de los experimentos reportados en el artículo de MARS.
- **Utilidades de análisis y visualización** — scripts para análisis de resultados, graficación, generación de figuras y construcción de cargas de trabajo.

---

## Agradecimientos

Este repositorio se basa en los siguientes proyectos de código abierto:

- [vLLM](https://github.com/vllm-project/vllm)
- [vLLM-continuum](https://github.com/Hanchenli/vllm-continuum)
- [OpenHands](https://github.com/All-Hands-AI/OpenHands)
- [GitTaskBench](https://github.com/git-task-bench/GitTaskBench)
- [Terminal-Bench](https://github.com/laude-institute/terminal-bench)

Agradecemos a los autores y colaboradores de estos proyectos por su valioso trabajo.
