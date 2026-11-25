#!/usr/bin/env bash
#
# Build a self-contained CAT installation bundle.
#
# This script is intended to run on a machine that can access GitHub and the
# public source archives. The generated bundle is then copied to the target
# server, where install_cat_bundle.sh installs CAT without running git clone.

set -Eeuo pipefail

# macOS bsdtar/libarchive writes extended attributes such as
# LIBARCHIVE.xattr.com.apple.provenance by default. Linux tar can safely ignore
# them, but the warning is noisy, so disable Apple metadata at archive creation.
export COPYFILE_DISABLE=1
export LC_ALL="${LC_ALL:-C}"
export LANG="${LANG:-C}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${CAT_REPO_ROOT:-}" ]; then
  CAT_REPO_ROOT="$(cd "$CAT_REPO_ROOT" && pwd)"
else
  CAT_REPO_ROOT=""
  for candidate in "$SCRIPT_DIR" "${SCRIPT_DIR}/.." "$(pwd)"; do
    if [ -f "${candidate}/setup.py" ] && [ -f "${candidate}/requirements.txt" ] && [ -d "${candidate}/src/ctc_crf" ]; then
      CAT_REPO_ROOT="$(cd "$candidate" && pwd)"
      break
    fi
  done
  if [ -z "$CAT_REPO_ROOT" ]; then
    printf '[cat-bundle][error] cannot infer CAT_REPO_ROOT; set CAT_REPO_ROOT=/path/to/CAT\n' >&2
    exit 1
  fi
fi
OUT_DIR="${OUT_DIR:-$(pwd)}"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
INSTALL_SCRIPT="${INSTALL_SCRIPT:-${SCRIPT_DIR}/install_cat_bundle.sh}"
BUNDLE_README="${BUNDLE_README:-${SCRIPT_DIR}/README_cat_bundle.md}"
BUNDLE_NAME="${BUNDLE_NAME:-cat-install-bundle-$(date +%Y%m%d-%H%M%S)}"
WORK_DIR="${OUT_DIR}/${BUNDLE_NAME}"
PAYLOAD_DIR="${WORK_DIR}/payload"
FETCH_DEPS="${FETCH_DEPS:-1}"
DOWNLOAD_PY_WHEELS="${DOWNLOAD_PY_WHEELS:-0}"
STRIP_GIT_METADATA="${STRIP_GIT_METADATA:-0}"
LATEST_TORCH_TRANSFORMERS="${LATEST_TORCH_TRANSFORMERS:-0}"
if [ -n "${TRANSFORMERS_SPEC:-}" ]; then
  TRANSFORMERS_REQUIREMENT="$TRANSFORMERS_SPEC"
elif [ "$LATEST_TORCH_TRANSFORMERS" = "1" ]; then
  TRANSFORMERS_REQUIREMENT="transformers"
else
  TRANSFORMERS_REQUIREMENT="transformers>=4.12.3,<4.54"
fi
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
TAR_CREATE_FLAGS=()

log() {
  printf '[cat-bundle] %s\n' "$*"
}

die() {
  printf '[cat-bundle][error] %s\n' "$*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

init_tar_flags() {
  # bsdtar on macOS needs explicit flags in addition to COPYFILE_DISABLE to
  # avoid PAX xattr records that GNU tar reports as unknown extended headers.
  # Probe each option so the same script remains usable with GNU tar on Linux.
  local opt
  for opt in --disable-copyfile --no-xattrs --no-acls; do
    if COPYFILE_DISABLE=1 tar "$opt" -cf "/tmp/cat-tar-option-test-$$.tar" --files-from /dev/null >/dev/null 2>&1; then
      TAR_CREATE_FLAGS+=("$opt")
    fi
  done
  rm -f "/tmp/cat-tar-option-test-$$.tar"
}

tar_create() {
  COPYFILE_DISABLE=1 tar "${TAR_CREATE_FLAGS[@]}" "$@"
}

download_file() {
  # Keep the archive filename stable because several upstream build scripts look
  # for exact names such as openfst-1.6.7.tar.gz before falling back to network.
  local url="$1"
  local dest="$2"
  mkdir -p "$(dirname "$dest")"
  if [ -s "$dest" ]; then
    log "reuse archive: $dest"
    return 0
  fi
  log "download: $url"
  curl -L --retry 3 --connect-timeout 20 -o "${dest}.tmp" "$url"
  mv "${dest}.tmp" "$dest"
}

clone_repo() {
  local name="$1"
  local url="$2"
  local ref="$3"
  local recursive="$4"
  local dest="${PAYLOAD_DIR}/third_party/repos/${name}"

  if [ "$FETCH_DEPS" != "1" ]; then
    log "skip fetch repo because FETCH_DEPS=0: $name"
    return 0
  fi

  if [ -d "$dest/.git" ]; then
    log "refresh repo: $name"
    git -C "$dest" fetch --tags --prune origin
  elif [ -d "$dest" ]; then
    die "$dest exists but is not a git repository; remove it or choose another BUNDLE_NAME"
  else
    log "clone repo: $name"
    if [ "$recursive" = "1" ]; then
      git clone --recursive "$url" "$dest"
    else
      git clone "$url" "$dest"
    fi
  fi

  if ! git -C "$dest" checkout "$ref"; then
    local default_branch
    default_branch="$(git -C "$dest" symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | sed 's#^origin/##')"
    [ -n "$default_branch" ] || die "cannot checkout $name ref '$ref' and cannot detect origin/HEAD"
    log "fallback to ${name} default branch: ${default_branch}"
    git -C "$dest" checkout "$default_branch"
  fi
  if [ "$recursive" = "1" ]; then
    git -C "$dest" submodule update --init --recursive
  fi
}

write_runtime_requirements() {
  local src_req="${CAT_REPO_ROOT}/requirements.txt"
  local dst_dir="${PAYLOAD_DIR}/requirements"
  local dst_req="${dst_dir}/requirements.runtime.txt"
  mkdir -p "$dst_dir"
  [ -f "$src_req" ] || die "requirements.txt not found at $src_req"

  # The original requirements.txt contains editable git URLs. Those are replaced
  # by local source trees in payload/third_party/repos, so the target installer
  # should only ask pip mirrors for ordinary Python packages. Torch is installed
  # through conda so CUDA variants stay explicit and reproducible.
  awk -v transformers_requirement="$TRANSFORMERS_REQUIREMENT" '
    /^[[:space:]]*$/ { print; next }
    /^[[:space:]]*#/ { print; next }
    /^[[:space:]]*-e[[:space:]]+git\+/ { next }
    /^[[:space:]]*torch([<>=!~ ]|$)/ { next }
    /^[[:space:]]*jiwer([<>=!~ ]|$)/ { print "jiwer>=2.2.0,<3.0"; next }
    /^[[:space:]]*transformers([<>=!~ ]|$)/ { print transformers_requirement; next }
    { print }
  ' "$src_req" >"$dst_req"

  cat >"${dst_dir}/requirements.local-sources.txt" <<'REQ'
# Copied from payload/third_party/repos to CAT/src, then installed by install_cat_bundle.sh.
torch-gather
warp-rnnt
webdataset
warp-ctct
ctc-align-cuda
REQ
}

copy_cat_source() {
  local cat_dst="${PAYLOAD_DIR}/source/CAT"
  mkdir -p "$cat_dst"
  log "copy CAT source tree"

  # Use tar instead of git archive so user-local, currently untracked source
  # changes such as cat_github/ are carried into the bundle. Exclude transient
  # outputs and VCS metadata to keep the install package portable.
  local excludes=(
    --exclude='./.git'
    --exclude='./outputs'
    --exclude='./.DS_Store'
    --exclude='*/.DS_Store'
    --exclude='*/__pycache__'
    --exclude='*.pyc'
    --exclude='*.pyo'
  )

  # The build script can write bundles outside outputs/ or even under the
  # repository root. If the staging directory is inside CAT_REPO_ROOT, exclude it
  # from the source snapshot so the bundle never recursively contains itself.
  if [[ "${WORK_DIR}/" == "${CAT_REPO_ROOT}/"* ]]; then
    local rel_work_dir="${WORK_DIR#${CAT_REPO_ROOT}/}"
    excludes+=(--exclude="./${rel_work_dir}")
  fi

  tar_create -C "$CAT_REPO_ROOT" "${excludes[@]}" -cf - . | tar -C "$cat_dst" -xf -

  # Some working copies in this project family temporarily keep the package
  # under cat_github/. CAT imports and setup metadata expect cat/, so normalize
  # only inside the staging area and leave the original checkout untouched.
  if [ ! -d "${cat_dst}/cat" ] && [ -d "${cat_dst}/cat_github" ]; then
    log "normalize staged package directory: cat_github -> cat"
    mv "${cat_dst}/cat_github" "${cat_dst}/cat"
  fi
}

write_manifest() {
  local manifest="${PAYLOAD_DIR}/MANIFEST.txt"
  {
    printf 'CAT bundle generated at: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'CAT_REPO_ROOT: %s\n' "$CAT_REPO_ROOT"
    printf 'INSTALL_SCRIPT: %s\n' "$INSTALL_SCRIPT"
    printf 'BUNDLE_README: %s\n' "$BUNDLE_README"
    printf 'LATEST_TORCH_TRANSFORMERS: %s\n' "$LATEST_TORCH_TRANSFORMERS"
    printf 'TRANSFORMERS_REQUIREMENT: %s\n' "$TRANSFORMERS_REQUIREMENT"
    printf 'BUNDLE_NAME: %s\n' "$BUNDLE_NAME"
    printf '\nBundled source repositories:\n'
    for repo in "${PAYLOAD_DIR}"/third_party/repos/*; do
      [ -d "$repo" ] || continue
      printf -- '- %s\n' "$(basename "$repo")"
    done
    printf '\nBundled source archives:\n'
    for archive in "${PAYLOAD_DIR}"/third_party/tarballs/*; do
      [ -f "$archive" ] || continue
      printf -- '- %s\n' "$(basename "$archive")"
    done
  } >"$manifest"
}

strip_git_metadata() {
  if [ "$STRIP_GIT_METADATA" != "1" ]; then
    return 0
  fi
  # The target installer never needs git metadata. Removing it proves the target
  # side cannot accidentally depend on git clone/fetch, and it shrinks transfer
  # size for release artifacts.
  find "${PAYLOAD_DIR}/third_party/repos" -name .git -type d -prune -exec rm -rf {} +
}

download_wheelhouse_if_requested() {
  if [ "$DOWNLOAD_PY_WHEELS" != "1" ]; then
    return 0
  fi
  need_cmd python
  log "download Python wheelhouse for runtime requirements"
  mkdir -p "${PAYLOAD_DIR}/wheelhouse"
  python -m pip download \
    -i "$PIP_INDEX_URL" \
    -r "${PAYLOAD_DIR}/requirements/requirements.runtime.txt" \
    -d "${PAYLOAD_DIR}/wheelhouse"
}

main() {
  need_cmd tar
  need_cmd awk
  need_cmd curl
  need_cmd git
  init_tar_flags

  [ -f "${CAT_REPO_ROOT}/setup.py" ] || die "setup.py not found under CAT_REPO_ROOT=$CAT_REPO_ROOT"
  mkdir -p "${PAYLOAD_DIR}/third_party/repos" "${PAYLOAD_DIR}/third_party/tarballs"

  copy_cat_source
  write_runtime_requirements

  clone_repo ctcdecode https://github.com/WayenVan/ctcdecode.git master 1
  clone_repo kenlm https://github.com/kpu/kenlm.git master 0
  clone_repo Phonetisaurus https://github.com/AdolfVonKleist/Phonetisaurus.git master 0
  clone_repo torch-gather https://github.com/maxwellzh/torch-gather.git main 0
  clone_repo warp-rnnt https://github.com/maxwellzh/warp-rnnt.git dev 0
  clone_repo webdataset https://github.com/webdataset/webdataset.git d7334016f44a03c4a385971aa835c4f460d3f30a 0
  clone_repo warp-ctct https://github.com/maxwellzh/warp-ctct.git main 0
  clone_repo ctc-align-cuda https://github.com/maxwellzh/ctc-align-cuda.git main 0

  if [ "$FETCH_DEPS" = "1" ]; then
    download_file http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.7.tar.gz \
      "${PAYLOAD_DIR}/third_party/tarballs/openfst-1.6.7.tar.gz"
    download_file http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.7.2.tar.gz \
      "${PAYLOAD_DIR}/third_party/tarballs/openfst-1.7.2.tar.gz"
    download_file https://archives.boost.io/release/1.67.0/source/boost_1_67_0.tar.gz \
      "${PAYLOAD_DIR}/third_party/tarballs/boost_1_67_0.tar.gz"
  fi

  download_wheelhouse_if_requested
  strip_git_metadata

  need_file "$INSTALL_SCRIPT"
  need_file "$BUNDLE_README"
  cp "$INSTALL_SCRIPT" "${WORK_DIR}/install_cat_bundle.sh"
  cp "$BUNDLE_README" "${WORK_DIR}/README.md"
  chmod +x "${WORK_DIR}/install_cat_bundle.sh"
  write_manifest

  log "create archive: ${OUT_DIR}/${BUNDLE_NAME}.tar.gz"
  tar_create -C "$OUT_DIR" -czf "${OUT_DIR}/${BUNDLE_NAME}.tar.gz" "$BUNDLE_NAME"
  log "done: ${OUT_DIR}/${BUNDLE_NAME}.tar.gz"
}

main "$@"
