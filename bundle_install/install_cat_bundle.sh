#!/usr/bin/env bash
#
# Install CAT from a prepared bundle without git clone.
#
# The script starts from conda bootstrap, creates a CUDA-ready Python
# environment, installs mirror-friendly Python dependencies, rebuilds native
# CAT dependencies from local source trees, and finally installs CAT itself.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="${BUNDLE_ROOT:-$SCRIPT_DIR}"
PAYLOAD_DIR="${PAYLOAD_DIR:-${BUNDLE_ROOT}/payload}"

CONDA_HOME="${CONDA_HOME:-${HOME}/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-cat}"
RECREATE_CONDA_ENV="${RECREATE_CONDA_ENV:-0}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CUDA_VERSION="${CUDA_VERSION:-12.1}"
TORCH_VERSION="${TORCH_VERSION:-2.1.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.16.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.1.0}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"
LATEST_TORCH_TRANSFORMERS="${LATEST_TORCH_TRANSFORMERS:-0}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-}"
DEFAULT_TRANSFORMERS_SPEC="${DEFAULT_TRANSFORMERS_SPEC:-transformers>=4.12.3,<4.54}"
INSTALL_CUDA_BUILD_TOOLKIT="${INSTALL_CUDA_BUILD_TOOLKIT:-0}"

CAT_INSTALL_ROOT="${CAT_INSTALL_ROOT:-${HOME}/CAT}"
OVERWRITE_CAT="${OVERWRITE_CAT:-0}"
BUILD_CTCDECODE="${BUILD_CTCDECODE:-1}"
BUILD_KENLM="${BUILD_KENLM:-1}"
BUILD_CTC_CRF="${BUILD_CTC_CRF:-1}"
PATCH_CTC_CRF_CUDA_ARCH="${PATCH_CTC_CRF_CUDA_ARCH:-1}"
CTC_CRF_CUDA_ARCH_LIST="${CTC_CRF_CUDA_ARCH_LIST:-auto}"
INSTALL_LOCAL_SOURCE_DEPS="${INSTALL_LOCAL_SOURCE_DEPS:-1}"
EDITABLE_LOCAL_SOURCE_DEPS="${EDITABLE_LOCAL_SOURCE_DEPS:-1}"
BUILD_G2P="${BUILD_G2P:-1}"
BUILD_FST_DECODER="${BUILD_FST_DECODER:-1}"

CONDA_MIRROR_BASE="${CONDA_MIRROR_BASE:-https://mirrors.tuna.tsinghua.edu.cn/anaconda}"
PYTORCH_CONDA_CHANNEL="${PYTORCH_CONDA_CHANNEL:-${CONDA_MIRROR_BASE}/cloud/pytorch}"
NVIDIA_CONDA_CHANNEL="${NVIDIA_CONDA_CHANNEL:-https://conda.anaconda.org/nvidia}"
CONDA_FORGE_CHANNEL="${CONDA_FORGE_CHANNEL:-${CONDA_MIRROR_BASE}/cloud/conda-forge}"
CONDA_PKGS_MAIN_CHANNEL="${CONDA_PKGS_MAIN_CHANNEL:-${CONDA_MIRROR_BASE}/pkgs/main}"
CONDA_PKGS_R_CHANNEL="${CONDA_PKGS_R_CHANNEL:-${CONDA_MIRROR_BASE}/pkgs/r}"
MINICONDA_URL="${MINICONDA_URL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
PIP_OFFLINE="${PIP_OFFLINE:-0}"
AUTO_ACTIVATE_CAT="${AUTO_ACTIVATE_CAT:-1}"
WRITE_SHELL_RC="${WRITE_SHELL_RC:-1}"
SHELL_RC_FILE="${SHELL_RC_FILE:-}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

CAT_SRC="${PAYLOAD_DIR}/source/CAT"
REPOS_DIR="${PAYLOAD_DIR}/third_party/repos"
TARBALL_DIR="${PAYLOAD_DIR}/third_party/tarballs"
WHEELHOUSE_DIR="${PAYLOAD_DIR}/wheelhouse"
STATE_DIR="${CAT_INSTALL_ROOT}/.cat_install_state"
REQUESTED_MODULES=()
DONE_RUNTIME_DEPS=0
DONE_LOCAL_SOURCE_DEPS=0
DONE_CTCDECODE=0
DONE_KENLM=0
DONE_CTC_CRF=0
DONE_FST_DECODER=0
DONE_G2P=0
DONE_CAT_PACKAGE=0

log() {
  printf '[cat-install] %s\n' "$*"
}

die() {
  printf '[cat-install][error] %s\n' "$*" >&2
  exit 1
}

warn_latest_torch_transformers_profile() {
  if [ "$LATEST_TORCH_TRANSFORMERS" != "1" ]; then
    return 0
  fi

  log "latest torch/transformers profile enabled"
  log "this profile is not the default verified CAT stack; native extensions such as ctc-crf, warp-rnnt, warp-ctct and ctc-align-cuda must be rebuilt and smoke-tested after installation"
}

run() {
  log "+ $*"
  "$@"
}

need_file() {
  [ -f "$1" ] || die "missing required file: $1"
}

need_dir() {
  [ -d "$1" ] || die "missing required directory: $1"
}

conda_cmd() {
  # conda activation hooks are not nounset-clean. In particular MKL's
  # deactivate hook can read unset backup variables when the parent shell uses
  # `set -u`, so temporarily relax nounset around every conda entry point.
  set +eu
  conda "$@"
  local status=$?
  set -eu
  return "$status"
}

source_conda_profile() {
  set +eu
  # shellcheck disable=SC1091
  source "${CONDA_HOME}/etc/profile.d/conda.sh"
  local status=$?
  set -eu
  return "$status"
}

activate_conda_env() {
  set +eu
  conda activate "$1"
  local status=$?
  set -eu
  return "$status"
}

usage() {
  cat <<'USAGE'
Usage:
  bash install_cat_bundle.sh [all|cat|ctcdecode|kenlm|ctc-crf|fst-decoder|g2p-tool ...]

Default module is "cat", matching the original install.sh behavior:
  cat => ctcdecode + kenlm + ctc-crf + Python requirements + CAT package
  all => cat + fst-decoder + g2p-tool

Environment examples:
  CONDA_HOME=/home/tasi/miniconda3 bash install_cat_bundle.sh cat
  KALDI_ROOT=/opt/kaldi bash install_cat_bundle.sh fst-decoder
  bash install_cat_bundle.sh g2p-tool
USAGE
}

parse_args() {
  if [ "$#" -eq 0 ]; then
    REQUESTED_MODULES=(cat)
    return 0
  fi

  local arg
  for arg in "$@"; do
    case "$arg" in
      -h|--help)
        usage
        exit 0
        ;;
      -r|--remove)
        die "remove mode is not supported by the offline bundle installer"
        ;;
      -f|--force)
        OVERWRITE_CAT=1
        ;;
      all|cat|ctcdecode|kenlm|ctc-crf|fst-decoder|g2p-tool)
        REQUESTED_MODULES+=("$arg")
        ;;
      *)
        usage >&2
        die "unknown module or option: $arg"
        ;;
    esac
  done

  if [ "${#REQUESTED_MODULES[@]}" -eq 0 ]; then
    REQUESTED_MODULES=(cat)
  fi
}

module_requested() {
  local target="$1"
  local module
  for module in "${REQUESTED_MODULES[@]}"; do
    if [ "$module" = "all" ] || [ "$module" = "$target" ]; then
      return 0
    fi
    case "$module:$target" in
      cat:ctcdecode|cat:kenlm|cat:ctc-crf|cat:cat)
        return 0
        ;;
    esac
  done
  return 1
}

need_torch_runtime() {
  module_requested cat || module_requested ctcdecode || module_requested ctc-crf
}

need_cuda_build() {
  if module_requested ctc-crf && [ "$BUILD_CTC_CRF" = "1" ]; then
    return 0
  fi
  if module_requested cat && [ "$INSTALL_LOCAL_SOURCE_DEPS" = "1" ]; then
    return 0
  fi
  return 1
}

conda_channel_args() {
  printf '%s\n' \
    --override-channels \
    -c "${PYTORCH_CONDA_CHANNEL}" \
    -c "${NVIDIA_CONDA_CHANNEL}" \
    -c "${CONDA_FORGE_CHANNEL}" \
    -c "${CONDA_PKGS_MAIN_CHANNEL}" \
    -c "${CONDA_PKGS_R_CHANNEL}"
}

conda_env_exists() {
  conda_cmd env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"
}

conda_env_ready() {
  conda_cmd run -n "$CONDA_ENV_NAME" python -c \
    "import sys; req=tuple(map(int, '${PYTHON_VERSION}'.split('.')[:2])); raise SystemExit(0 if sys.version_info[:len(req)] == req else 1)"
}

create_conda_env() {
  local channels=("$@")
  log "create conda environment: ${CONDA_ENV_NAME} / python=${PYTHON_VERSION}"
  conda_cmd create -y "${channels[@]}" -n "$CONDA_ENV_NAME" "python=${PYTHON_VERSION}"
  conda_env_ready || die "conda environment was created but failed validation: ${CONDA_ENV_NAME}"
}

pip_install() {
  # Keep all pip calls behind one wrapper so mirror settings, local wheelhouse
  # fallback, and non-isolated native builds stay consistent across packages.
  local args=(install --no-build-isolation)
  if [ -d "$WHEELHOUSE_DIR" ]; then
    args+=(--find-links "$WHEELHOUSE_DIR")
  fi
  if [ "$PIP_OFFLINE" = "1" ]; then
    args+=(--no-index)
  else
    args+=(-i "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST")
  fi
  python -m pip "${args[@]}" "$@"
}

ensure_conda() {
  if [ -x "${CONDA_HOME}/bin/conda" ]; then
    log "reuse conda: ${CONDA_HOME}"
  elif command -v conda >/dev/null 2>&1; then
    local detected_conda_home
    detected_conda_home="$(conda info --base 2>/dev/null || true)"
    if [ -n "$detected_conda_home" ] && [ -x "${detected_conda_home}/bin/conda" ]; then
      CONDA_HOME="$detected_conda_home"
      log "reuse conda from PATH: ${CONDA_HOME}"
    else
      die "conda is in PATH but its base directory could not be detected; set CONDA_HOME explicitly"
    fi
  else
    log "install Miniconda: ${CONDA_HOME}"
    mkdir -p "$(dirname "$CONDA_HOME")"
    local installer="/tmp/miniconda-cat-install.sh"
    curl -L --retry 3 --connect-timeout 20 -o "$installer" "$MINICONDA_URL"
    bash "$installer" -b -p "$CONDA_HOME"
  fi

  source_conda_profile
}

ensure_conda_env() {
  local channels=()
  while IFS= read -r item; do
    channels+=("$item")
  done < <(conda_channel_args)

  if conda_env_exists; then
    if conda_env_ready; then
      log "reuse completed conda environment: ${CONDA_ENV_NAME}"
    elif [ "$RECREATE_CONDA_ENV" = "1" ]; then
      log "remove incomplete conda environment: ${CONDA_ENV_NAME}"
      conda_cmd env remove -y -n "$CONDA_ENV_NAME"
      create_conda_env "${channels[@]}"
    else
      die "conda environment exists but failed validation: ${CONDA_ENV_NAME}. Rerun with RECREATE_CONDA_ENV=1 to rebuild it."
    fi
  else
    create_conda_env "${channels[@]}"
  fi

  activate_conda_env "$CONDA_ENV_NAME"
  log "active python: $(command -v python)"

  # Native CAT extensions need a normal build toolchain. Prefer conda packages
  # here so a fresh server does not have to start with apt/yum setup.
  conda_cmd install -y "${channels[@]}" \
    "cmake<4" make ninja pkg-config automake autoconf libtool curl wget sox git \
    boost-cpp eigen zlib bzip2 xz

  if [ "$INSTALL_TORCH" = "1" ]; then
    if [ "$LATEST_TORCH_TRANSFORMERS" = "1" ]; then
      log "install latest PyTorch from conda channels"
      conda_cmd install -y "${channels[@]}" \
        pytorch torchvision torchaudio \
        "pytorch-cuda=${CUDA_VERSION}"
    elif need_torch_runtime; then
      log "install pinned PyTorch from conda channels"
      conda_cmd install -y "${channels[@]}" \
        "pytorch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}" \
        "pytorch-cuda=${CUDA_VERSION}"
    fi
  fi

  if [ "$INSTALL_CUDA_BUILD_TOOLKIT" = "1" ]; then
    log "install CUDA build toolkit from conda mirror"
    conda_cmd install -y "${channels[@]}" \
      "cuda-nvcc=${CUDA_VERSION}" \
      "cuda-cudart-dev=${CUDA_VERSION}"
  fi

  pip_install --upgrade "pip<25" "setuptools==69.5.1" wheel
  pip_install "numpy==1.26.4"
}

validate_cuda_build_environment() {
  # CAT's CTC/RNN-T/CTC-align extensions include CUDA code. PyTorch's conda
  # runtime package is not always enough to compile those extensions; the target
  # machine needs either a system CUDA toolkit or INSTALL_CUDA_BUILD_TOOLKIT=1.
  if ! need_cuda_build; then
    return 0
  fi

  if [ -x "${CONDA_PREFIX:-}/bin/nvcc" ]; then
    export CUDA_HOME="$CONDA_PREFIX"
  elif command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
  elif [ -x /usr/local/cuda/bin/nvcc ]; then
    export CUDA_HOME=/usr/local/cuda
    export PATH="${CUDA_HOME}/bin:${PATH}"
  else
    die "nvcc not found. Install a CUDA toolkit on the server, or rerun with INSTALL_CUDA_BUILD_TOOLKIT=1."
  fi

  export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  log "CUDA_HOME=${CUDA_HOME}"
}

prepare_cat_tree() {
  need_dir "$CAT_SRC"
  mkdir -p "$(dirname "$CAT_INSTALL_ROOT")"

  if [ -e "$CAT_INSTALL_ROOT" ] && [ "$OVERWRITE_CAT" != "1" ]; then
    log "reuse existing CAT_INSTALL_ROOT: $CAT_INSTALL_ROOT"
  else
    if [ -e "$CAT_INSTALL_ROOT" ]; then
      local backup="${CAT_INSTALL_ROOT}.backup.$(date +%Y%m%d-%H%M%S)"
      log "move existing CAT tree to backup: $backup"
      mv "$CAT_INSTALL_ROOT" "$backup"
    fi
    log "copy CAT source to: $CAT_INSTALL_ROOT"
    cp -a "$CAT_SRC" "$CAT_INSTALL_ROOT"
  fi

  if [ ! -d "${CAT_INSTALL_ROOT}/cat" ] && [ -d "${CAT_INSTALL_ROOT}/cat_github" ]; then
    log "normalize installed package directory: cat_github -> cat"
    mv "${CAT_INSTALL_ROOT}/cat_github" "${CAT_INSTALL_ROOT}/cat"
  fi
  [ -d "${CAT_INSTALL_ROOT}/cat" ] || die "CAT package directory is missing: ${CAT_INSTALL_ROOT}/cat"
  mkdir -p "${CAT_INSTALL_ROOT}/src/bin" "$STATE_DIR"
}

install_runtime_python_deps() {
  local runtime_deps_were_skipped=0
  if [ "$DONE_RUNTIME_DEPS" = "0" ]; then
    DONE_RUNTIME_DEPS=1
    need_file "${PAYLOAD_DIR}/requirements/requirements.runtime.txt"
    log "install ordinary Python dependencies"
    pip_install -r "${PAYLOAD_DIR}/requirements/requirements.runtime.txt"
  else
    runtime_deps_were_skipped=1
    log "runtime Python dependencies already marked done; checking transformers profile"
  fi

  if [ "$LATEST_TORCH_TRANSFORMERS" = "1" ]; then
    local transformers_spec="${TRANSFORMERS_SPEC:-transformers}"
    log "upgrade transformers for latest torch/transformers profile: ${transformers_spec}"
    pip_install --upgrade "$transformers_spec"
  elif [ -n "$TRANSFORMERS_SPEC" ]; then
    log "override transformers package: ${TRANSFORMERS_SPEC}"
    pip_install --upgrade "$TRANSFORMERS_SPEC"
  elif [ "$runtime_deps_were_skipped" = "1" ]; then
    # A previous bundle run may have installed a newer transformers release
    # before the default range was tightened for torch 2.1. Re-apply the default
    # profile when runtime deps are skipped so rerunning the installer repairs
    # an existing cat environment instead of leaving the old incompatible wheel.
    log "ensure default transformers compatibility: ${DEFAULT_TRANSFORMERS_SPEC}"
    pip_install --upgrade "$DEFAULT_TRANSFORMERS_SPEC"
  fi
}

copy_repo_to_cat_src() {
  local repo_name="$1"
  local dst_name="${2:-$1}"
  local src="${REPOS_DIR}/${repo_name}"
  local dst="${CAT_INSTALL_ROOT}/src/${dst_name}"
  need_dir "$src"
  if [ ! -d "$dst" ]; then
    log "copy local source repo: ${repo_name} -> src/${dst_name}"
    cp -a "$src" "$dst"
  else
    log "reuse local source repo: $dst"
  fi
}

install_local_editable_repos() {
  [ "$DONE_LOCAL_SOURCE_DEPS" = "0" ] || return 0
  DONE_LOCAL_SOURCE_DEPS=1
  [ "$INSTALL_LOCAL_SOURCE_DEPS" = "1" ] || return 0
  local repos=(torch-gather warp-rnnt webdataset warp-ctct ctc-align-cuda)
  local repo
  for repo in "${repos[@]}"; do
    copy_repo_to_cat_src "$repo"
    local cat_src_repo="${CAT_INSTALL_ROOT}/src/${repo}"
    need_dir "$cat_src_repo"
    if [ "$EDITABLE_LOCAL_SOURCE_DEPS" = "1" ]; then
      log "install local editable source: $repo"
      if ! pip_install -e "$cat_src_repo"; then
        die "editable install failed for ${repo}. If this is caused by pip 25+ legacy editable changes, rerun with EDITABLE_LOCAL_SOURCE_DEPS=0 to install local source dependencies in non-editable mode."
      fi
    else
      log "install local source package: $repo"
      pip_install "$cat_src_repo"
    fi
  done
}

install_ctcdecode() {
  [ "$DONE_CTCDECODE" = "0" ] || return 0
  DONE_CTCDECODE=1
  [ "$BUILD_CTCDECODE" = "1" ] || return 0
  copy_repo_to_cat_src ctcdecode ctcdecode

  # ctcdecode historically downloads OpenFST/Boost during setup. Put the
  # archives in several likely lookup locations so the build can stay offline
  # even if upstream helper scripts only check relative paths.
  local ctcdecode_dir="${CAT_INSTALL_ROOT}/src/ctcdecode"
  mkdir -p "${ctcdecode_dir}/third_party" "${ctcdecode_dir}/third_party/downloads"
  for archive in openfst-1.6.7.tar.gz boost_1_67_0.tar.gz; do
    if [ -f "${TARBALL_DIR}/${archive}" ]; then
      cp -f "${TARBALL_DIR}/${archive}" "${ctcdecode_dir}/${archive}"
      cp -f "${TARBALL_DIR}/${archive}" "${ctcdecode_dir}/third_party/${archive}"
      cp -f "${TARBALL_DIR}/${archive}" "${ctcdecode_dir}/third_party/downloads/${archive}"
    fi
  done

  log "install ctcdecode from local source"
  pip_install "${ctcdecode_dir}"
}

install_kenlm() {
  [ "$DONE_KENLM" = "0" ] || return 0
  DONE_KENLM=1
  [ "$BUILD_KENLM" = "1" ] || return 0
  copy_repo_to_cat_src kenlm kenlm
  local kenlm_dir="${CAT_INSTALL_ROOT}/src/kenlm"

  log "install kenlm Python binding"
  pip_install -e "$kenlm_dir"

  log "build kenlm binaries"
  mkdir -p "${kenlm_dir}/build"
  (
    cd "${kenlm_dir}/build"
    cmake ..
    make -j "$JOBS"
  )

  mkdir -p "${CAT_INSTALL_ROOT}/src/bin"
  (
    cd "${CAT_INSTALL_ROOT}/src/bin"
    ln -snf ../kenlm/build/bin/* ./
  )
}

detect_ctc_crf_cuda_arches() {
  # ctc-crf's CMake files carry a static, historical architecture list. That
  # list is enough for older cards and A100 (sm_80), but it does not know about
  # newer devices such as Ada (sm_89) or Hopper (sm_90). Detect the actual
  # visible devices through PyTorch after the conda environment is prepared, and
  # append only the missing architectures before running CMake.
  #
  # Accepted manual forms:
  #   CTC_CRF_CUDA_ARCH_LIST=90
  #   CTC_CRF_CUDA_ARCH_LIST="8.0 8.6 9.0"
  #   CTC_CRF_CUDA_ARCH_LIST="sm_80,sm_90"
  #
  # Set CTC_CRF_CUDA_ARCH_LIST=0 to disable auto-addition and keep only the
  # architecture flags shipped by upstream CAT after the sm_35 compatibility
  # patch below.
  if [ "$CTC_CRF_CUDA_ARCH_LIST" = "0" ]; then
    return 0
  fi

  python - "$CTC_CRF_CUDA_ARCH_LIST" <<'PY'
import re
import sys

raw = sys.argv[1].strip()
arches = set()

def add_arch(token: str) -> None:
    """Normalize CUDA arch tokens to the compact nvcc suffix form, e.g. 8.0 -> 80."""
    token = token.strip().lower()
    token = token.replace("sm_", "").replace("compute_", "")
    token = token.replace("+ptx", "")
    if not token:
        return
    if "." in token:
        major, minor = token.split(".", 1)
        token = f"{major}{minor[:1]}"
    if token.isdigit():
        arches.add(token)

if raw and raw.lower() not in {"auto", "1", "true", "yes"}:
    for item in re.split(r"[\s,;:]+", raw):
        add_arch(item)
else:
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and torch.cuda.is_available():
        for device_index in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(device_index)
            arches.add(f"{major}{minor}")

print(" ".join(sorted(arches, key=int)))
PY
}

patch_ctc_crf_setup_py() {
  local ctc_crf_dir="$1"
  local setup_py="${ctc_crf_dir}/setup.py"
  need_file "$setup_py"

  # PyTorch's BuildExtension already knows which C++ standard its headers need.
  # Upstream ctc-crf still hard-codes C++14, and that override can make the final
  # `python setup.py install` step fail with newer PyTorch headers. Remove the
  # stale override instead of hard-coding another standard here, so the active
  # PyTorch version remains the source of truth.
  #
  # The include path is patched at the same time: a server may use either
  # /usr/local/cuda or a conda-provided CUDA toolkit. `validate_cuda_build_environment`
  # exports CUDA_HOME before this function runs, so setup.py can follow it.
  local patched
  patched="$(python - "$setup_py" <<'PY'
import sys
from pathlib import Path

setup_py = Path(sys.argv[1])
text = setup_py.read_text()
original = text

text = text.replace("'-std=c++14',\n                                      ", "")
text = text.replace("'-std=c++14', ", "")
text = text.replace("'-std=c++14',", "")
text = text.replace("'-std=c++14'", "")
text = text.replace(
    "'-I/usr/local/cuda/include'",
    "'-I' + os.path.join(os.environ.get('CUDA_HOME', '/usr/local/cuda'), 'include')",
)

if text != original:
    setup_py.write_text(text)
    print("1")
else:
    print("0")
PY
)"

  if [ "$patched" = "1" ]; then
    log "patch ctc_crf setup.py for PyTorch-managed C++ standard/CUDA_HOME"
    rm -rf "${ctc_crf_dir}/build" "${ctc_crf_dir}/dist" "${ctc_crf_dir}/ctc_crf.egg-info"
  fi
}

patch_ctc_crf_cuda_arch_flags() {
  [ "$PATCH_CTC_CRF_CUDA_ARCH" = "1" ] || return 0

  local ctc_crf_dir="$1"
  local cuda_arches
  cuda_arches="$(detect_ctc_crf_cuda_arches)"
  local cmake_file
  local patched=0
  for cmake_file in \
    "${ctc_crf_dir}/gpu_ctc/CMakeLists.txt" \
    "${ctc_crf_dir}/gpu_den/CMakeLists.txt"; do
    [ -f "$cmake_file" ] || continue
    if grep -q '^set(CUDA_NVCC_FLAGS.*compute_35,code=sm_35' "$cmake_file"; then
      log "patch CUDA arch flags for CUDA>=12: $cmake_file"
      awk '
        $0 ~ /^set\(CUDA_NVCC_FLAGS.*compute_35,code=sm_35/ {
          print "IF (CUDA_VERSION VERSION_LESS \"12.0\")"
          print "    " $0
          print "ENDIF()"
          next
        }
        { print }
      ' "$cmake_file" >"${cmake_file}.tmp"
      mv "${cmake_file}.tmp" "$cmake_file"
      patched=1
    fi

    if [ -n "$cuda_arches" ]; then
      local cuda_arch
      for cuda_arch in $cuda_arches; do
        if ! grep -q "compute_${cuda_arch},code=sm_${cuda_arch}" "$cmake_file"; then
          log "add CUDA arch sm_${cuda_arch} for ctc_crf: $cmake_file"
          awk -v cuda_arch="$cuda_arch" '
            BEGIN { inserted = 0 }
            !inserted && $0 == "set(CUDA_NVCC_FLAGS \"${CUDA_NVCC_FLAGS}\")" {
              print "set(CUDA_NVCC_FLAGS \"${CUDA_NVCC_FLAGS} -gencode arch=compute_" cuda_arch ",code=sm_" cuda_arch "\")"
              inserted = 1
            }
            { print }
          ' "$cmake_file" >"${cmake_file}.tmp"
          mv "${cmake_file}.tmp" "$cmake_file"
          patched=1
        fi
      done
    fi
  done

  if [ "$patched" = "1" ]; then
    rm -rf "${ctc_crf_dir}/gpu_ctc/build" "${ctc_crf_dir}/gpu_den/build"
  fi
}

install_ctc_crf() {
  [ "$DONE_CTC_CRF" = "0" ] || return 0
  DONE_CTC_CRF=1
  [ "$BUILD_CTC_CRF" = "1" ] || return 0
  local ctc_crf_dir="${CAT_INSTALL_ROOT}/src/ctc_crf"
  need_dir "$ctc_crf_dir"
  need_file "${TARBALL_DIR}/openfst-1.6.7.tar.gz"
  cp -f "${TARBALL_DIR}/openfst-1.6.7.tar.gz" "${ctc_crf_dir}/openfst-1.6.7.tar.gz"
  patch_ctc_crf_setup_py "$ctc_crf_dir"
  patch_ctc_crf_cuda_arch_flags "$ctc_crf_dir"
  rm -rf "${ctc_crf_dir}/build" "${ctc_crf_dir}/dist" "${ctc_crf_dir}/ctc_crf.egg-info"

  log "build ctc_crf extension"
  (
    cd "$ctc_crf_dir"
    if command -v gcc-7 >/dev/null 2>&1 && command -v g++-7 >/dev/null 2>&1; then
      CC=gcc-7 CXX=g++-7 make
    elif command -v gcc-6 >/dev/null 2>&1 && command -v g++-6 >/dev/null 2>&1; then
      CC=gcc-6 CXX=g++-6 make
    else
      make
    fi
  )
}

install_fst_decoder() {
  [ "$DONE_FST_DECODER" = "0" ] || return 0
  DONE_FST_DECODER=1
  [ "$BUILD_FST_DECODER" = "1" ] || return 0
  [ -n "${KALDI_ROOT:-}" ] || die "BUILD_FST_DECODER=1 requires KALDI_ROOT"
  local decoder_dir="${CAT_INSTALL_ROOT}/src/fst-decoder"
  need_dir "$decoder_dir"

  log "build CAT fst decoder against KALDI_ROOT=${KALDI_ROOT}"
  (
    cd "$decoder_dir"
    KALDI_ROOT="$KALDI_ROOT" make
  )
  ln -snf ../fst-decoder/latgen-faster "${CAT_INSTALL_ROOT}/src/bin/latgen-faster"
}

install_g2p_tool() {
  [ "$DONE_G2P" = "0" ] || return 0
  DONE_G2P=1
  [ "$BUILD_G2P" = "1" ] || return 0
  local g2p_dir="${CAT_INSTALL_ROOT}/src/g2p-tool"
  local ofst_path="${g2p_dir}/openfst-1.7.2"
  need_dir "$g2p_dir"
  need_file "${TARBALL_DIR}/openfst-1.7.2.tar.gz"
  need_dir "${REPOS_DIR}/Phonetisaurus"

  log "build OpenFST 1.7.2 for Phonetisaurus"
  cp -f "${TARBALL_DIR}/openfst-1.7.2.tar.gz" "${g2p_dir}/openfst-1.7.2.tar.gz"
  if [ ! -f "${ofst_path}/.done" ]; then
    (
      cd "$g2p_dir"
      tar -zxf openfst-1.7.2.tar.gz
      rm -rf openfst-1.7.2-build
      mv openfst-1.7.2 openfst-1.7.2-build
      cd openfst-1.7.2-build
      ./configure --prefix="$ofst_path" --enable-static --enable-shared --enable-far --enable-ngram-fsts
      make -j "$JOBS"
      make install
      cd ..
      rm -rf openfst-1.7.2-build
      touch "${ofst_path}/.done"
    )
  fi

  if [ ! -d "${g2p_dir}/Phonetisaurus" ]; then
    cp -a "${REPOS_DIR}/Phonetisaurus" "${g2p_dir}/Phonetisaurus"
  fi

  log "build Phonetisaurus from local source"
  pip_install pybindgen
  (
    cd "${g2p_dir}/Phonetisaurus"
    if [ ! -f build/.done ]; then
      PYTHON=python ./configure --enable-python \
        --with-openfst-includes="${ofst_path}/include" \
        --with-openfst-libs="${ofst_path}/lib" \
        --prefix="$(pwd)/build"
      make -j "$JOBS"
      make install
      touch build/.done
    fi
    cd python
    cp ../.libs/Phonetisaurus.so ./
    python setup.py install
  )

  mkdir -p "${CAT_INSTALL_ROOT}/src/bin"
  (
    cd "${CAT_INSTALL_ROOT}/src/bin"
    ln -snf ../g2p-tool/Phonetisaurus/build/bin/* ./
  )
}

install_cat_package() {
  [ "$DONE_CAT_PACKAGE" = "0" ] || return 0
  DONE_CAT_PACKAGE=1
  log "install CAT package"
  (
    cd "$CAT_INSTALL_ROOT"
    pip_install -e .
  )
}

write_shell_hook() {
  [ "$WRITE_SHELL_RC" = "1" ] || return 0

  local rc_file="$SHELL_RC_FILE"
  if [ -z "$rc_file" ]; then
    case "${SHELL:-}" in
      */zsh) rc_file="${HOME}/.zshrc" ;;
      *) rc_file="${HOME}/.bashrc" ;;
    esac
  fi
  mkdir -p "$(dirname "$rc_file")"
  touch "$rc_file"

  local begin="# >>> cat conda environment >>>"
  local end="# <<< cat conda environment <<<"
  local tmp_file="${rc_file}.cat-tmp"

  awk -v begin="$begin" -v end="$end" '
    $0 == begin { skip = 1; next }
    $0 == end { skip = 0; next }
    skip != 1 { print }
  ' "$rc_file" >"$tmp_file"

  {
    printf '%s\n' "$begin"
    printf 'export CONDA_HOME=%q\n' "$CONDA_HOME"
    printf '[ -f "$CONDA_HOME/etc/profile.d/conda.sh" ] && . "$CONDA_HOME/etc/profile.d/conda.sh"\n'
    if [ "$AUTO_ACTIVATE_CAT" = "1" ]; then
      printf 'conda activate %q >/dev/null 2>&1 || true\n' "$CONDA_ENV_NAME"
    fi
    printf 'export CAT_ROOT=%q\n' "$CAT_INSTALL_ROOT"
    printf 'export PATH="$CAT_ROOT/src/bin:$PATH"\n'
    printf '%s\n' "$end"
  } >>"$tmp_file"

  mv "$tmp_file" "$rc_file"
  log "updated shell rc: $rc_file"
}

validate_install() {
  if module_requested cat; then
    log "validate CAT import"
    (
      cd /tmp
      python - <<'PY'
import importlib
for name in ["torch", "cat"]:
    importlib.import_module(name)
print("CAT import smoke test passed")
PY
    )
  fi

  if module_requested kenlm && [ "$BUILD_KENLM" = "1" ]; then
    [ -x "${CAT_INSTALL_ROOT}/src/bin/lmplz" ] || die "kenlm binary lmplz not found"
  fi
}

install_module() {
  local module="$1"
  case "$module" in
    ctcdecode)
      install_ctcdecode
      ;;
    kenlm)
      install_kenlm
      ;;
    ctc-crf)
      install_ctc_crf
      ;;
    cat)
      install_ctcdecode
      install_kenlm
      install_ctc_crf
      install_runtime_python_deps
      install_local_editable_repos
      install_cat_package
      ;;
    fst-decoder)
      install_fst_decoder
      ;;
    g2p-tool)
      install_g2p_tool
      ;;
    all)
      install_module cat
      install_fst_decoder
      install_g2p_tool
      ;;
    *)
      die "unknown module: $module"
      ;;
  esac
  log "installed module: $module"
}

install_requested_modules() {
  local module
  for module in "${REQUESTED_MODULES[@]}"; do
    install_module "$module"
  done
}

main() {
  parse_args "$@"
  warn_latest_torch_transformers_profile
  need_dir "$PAYLOAD_DIR"
  need_dir "$CAT_SRC"
  need_dir "$REPOS_DIR"
  need_dir "$TARBALL_DIR"

  ensure_conda
  ensure_conda_env
  validate_cuda_build_environment
  prepare_cat_tree
  install_requested_modules
  write_shell_hook
  validate_install

  log "done"
  log "CAT_ROOT=${CAT_INSTALL_ROOT}"
  log "conda activate ${CONDA_ENV_NAME}"
}

main "$@"
