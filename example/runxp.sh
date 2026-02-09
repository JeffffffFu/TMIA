#!/bin/bash
# 在脚本内 source conda 并 activate 指定环境。若本机有多个 Anaconda，可设置 CONDA_BASE 指定包含 RIA-LLM 的那个，例如：
#   export CONDA_BASE=/Wang-ds/jeff/software/anaconda3
#   ./example/runxp.sh
cd "$(dirname "$0")/.."

CONDA_ENV="${CONDA_ENV:-RIA-LLM}"
if [ -z "$CONDA_BASE" ] && command -v conda >/dev/null 2>&1; then
  CONDA_BASE=$(conda info --base 2>/dev/null)
fi
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  if conda env list 2>/dev/null | grep -q "^${CONDA_ENV}[[:space:]*]"; then
    conda activate "$CONDA_ENV"
    # 强制把当前环境的 bin 放在 PATH 最前，避免被其它 Anaconda 覆盖
    [ -n "$CONDA_PREFIX" ] && export PATH="$CONDA_PREFIX/bin:$PATH"
    echo "Using conda env: $CONDA_ENV (source + activate)"
  fi
fi

echo "Using Python: $(which python)"
python -c "import datasets" 2>/dev/null || {
  echo "Error: 'datasets' not found. Ensure conda env $CONDA_ENV exists and has: pip install datasets"
  exit 1
}
exec python main.py --attack_method TW_MIA --U_method continuous_update_finetune --dataset_name sst5 --net_name pythia70m --proportion_of_group_unlearn 0.01 --device cuda:0
