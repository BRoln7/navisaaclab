#!/usr/bin/env bash
# Use the launcher from the installed Isaac Lab checkout, or an active environment.
set -euo pipefail
# Make this checkout importable for scripts, including the local poselib tree.
# Scope this to the launcher process; do not modify the user's environment files.
SOCNAV_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SOCNAV_REPO_ROOT}/exts/socnav_sim:${SOCNAV_REPO_ROOT}/poselib:${SOCNAV_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "${1:-}" == "-p" || "${1:-}" == "--python" ]]; then
    shift
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        exec "${CONDA_PREFIX}/bin/python" "$@"
    elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
        exec "${VIRTUAL_ENV}/bin/python" "$@"
    else
        exec python "$@"
    fi
fi
if [[ -n "${ISAACLAB_ROOT:-}" && -f "${ISAACLAB_ROOT}/isaaclab.sh" ]]; then
    if [[ "$(realpath "${ISAACLAB_ROOT}/isaaclab.sh")" != "$(realpath "$0")" ]]; then
        exec "${ISAACLAB_ROOT}/isaaclab.sh" "$@"
    fi
fi
printf '%s\n' 'Activate an Isaac Lab 2.3 Python environment for -p, or set ISAACLAB_ROOT to the Isaac Lab checkout.' >&2
exit 1
