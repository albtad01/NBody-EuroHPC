#!/bin/bash
#SBATCH --account=EUHPC_TDEMO_26
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=nbody_gpu_record
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=nbody_gpu_record_%j.out
#SBATCH --error=nbody_gpu_record_%j.err

# Generate one visualization trajectory with the validated single-A100 backend.
# The Leonardo executable must already be built; this job never configures or builds it.
set -euo pipefail

module purge
module load profile/base
module load gcc/12.2.0
module load cuda/12.2

: "${SLURM_JOB_ID:?This script must be submitted with sbatch}"
: "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unavailable}"

ROOT="$(cd "$SLURM_SUBMIT_DIR" && pwd -P)"
BUILD="$ROOT/build-leonardo"
BIN="$BUILD/bin/murb"

git_root="$(git -C "$ROOT" rev-parse --show-toplevel)"
git_root="$(cd "$git_root" && pwd -P)"
[[ "$git_root" == "$ROOT" ]] || {
    echo "Submit this job from the Git worktree root: $ROOT" >&2
    exit 1
}
[[ -x "$BIN" && -f "$BUILD/murb-build.ready" ]] || {
    echo "Missing executable or successful build stamp: $BUILD. Build before submitting." >&2
    exit 1
}

revision="$(git -C "$ROOT" rev-parse HEAD)"
version="$("$BIN" --version)"
[[ "$version" == "murb revision=$revision dirty=0 "* ]] || {
    echo "Executable is stale or was built from dirty sources: $version" >&2
    exit 1
}
git -C "$ROOT" diff --quiet HEAD -- || {
    echo "Tracked sources changed after the build; rebuild before submitting." >&2
    exit 1
}
[[ "$version" == *" cuda=1 "* ]] || {
    echo "A CUDA-enabled build-leonardo/bin/murb is required: $version" >&2
    exit 1
}

N="${MURB_N:-10000}"
ITERS="${MURB_ITERS:-300}"
WARMUP="${MURB_WARMUP:-3}"
DT="${MURB_DT:-3600}"
RECORD_EVERY="${MURB_RECORD_EVERY:-2}"

require_positive_integer() {
    local name="$1"
    local value="$2"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
        echo "$name must be a positive integer: $value" >&2
        exit 1
    }
}

require_positive_integer MURB_N "$N"
require_positive_integer MURB_ITERS "$ITERS"
require_positive_integer MURB_WARMUP "$WARMUP"
require_positive_integer MURB_RECORD_EVERY "$RECORD_EVERY"
[[ "$RECORD_EVERY" -le "$ITERS" ]] || {
    echo "MURB_RECORD_EVERY must not exceed MURB_ITERS" >&2
    exit 1
}
awk -v value="$DT" 'BEGIN {
    valid = value ~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$/ && value + 0 > 0
    exit !valid
}' || {
    echo "MURB_DT must be a finite positive number: $DT" >&2
    exit 1
}

if [[ -n "${MURB_OUTPUT:-}" ]]; then
    output_requested="$MURB_OUTPUT"
else
    default_base=""
    if [[ -n "${SCRATCH:-}" && -d "$SCRATCH" ]]; then
        default_base="$SCRATCH"
    elif [[ -n "${WORK:-}" && -d "$WORK" ]]; then
        default_base="$WORK"
    else
        echo "Neither SCRATCH nor WORK is available; set MURB_OUTPUT to an absolute .murbtraj path." >&2
        exit 1
    fi
    output_requested="$default_base/murb-trajectories/a100-galaxy-${SLURM_JOB_ID}.murbtraj"
fi

[[ "$output_requested" == /* && "$output_requested" == *.murbtraj ]] || {
    echo "MURB_OUTPUT must be an absolute path ending in .murbtraj: $output_requested" >&2
    exit 1
}
output_directory="$(dirname -- "$output_requested")"
output_name="$(basename -- "$output_requested")"
mkdir -p "$output_directory"
output_directory="$(cd "$output_directory" && pwd -P)"
OUTPUT="$output_directory/$output_name"
case "$OUTPUT" in
    "$ROOT"|"$ROOT"/*)
        echo "Trajectory output must be outside the Git worktree: $OUTPUT" >&2
        exit 1
        ;;
esac
[[ ! -e "$OUTPUT" ]] || {
    echo "Refusing to overwrite existing trajectory: $OUTPUT" >&2
    exit 1
}

gpu_query="$(srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    --gpus-per-task=1 nvidia-smi --query-gpu=name --format=csv,noheader)"
gpu_models=()
while IFS= read -r model; do
    [[ -n "$model" ]] && gpu_models+=("$model")
done <<< "$gpu_query"
gpu_count="${#gpu_models[@]}"
[[ "$gpu_count" -eq 1 ]] || {
    echo "Expected exactly one visible GPU, found $gpu_count" >&2
    exit 1
}

expected_frames=$((ITERS / RECORD_EVERY))

echo "SLURM job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "Executable revision: $revision"
echo "GPU model: ${gpu_models[0]}"
echo "Visible GPU count: $gpu_count"
echo "N: $N"
echo "Iterations: $ITERS"
echo "Warm-up iterations: $WARMUP"
echo "Timestep: $DT"
echo "Recording stride: $RECORD_EVERY"
echo "Expected frame count: $expected_frames"
echo "Output path: $OUTPUT"
echo "$version"

export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OMP_PLACES=cores
export OMP_PROC_BIND=close

run_output="$(srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    --gpus-per-task=1 --cpu-bind=cores \
    "$BIN" -n "$N" -i "$ITERS" --warmup "$WARMUP" \
    --im gpu+tile+full --scheme galaxy --nv --dt "$DT" \
    --record "$OUTPUT" --record-every "$RECORD_EVERY")"
printf '%s\n' "$run_output"

summary=""
while IFS= read -r line; do
    [[ "$line" == completed_iterations=* ]] && summary="$line"
done <<< "$run_output"
[[ "$summary" =~ (^|[[:space:]])completed_iterations=([0-9]+) ]] || {
    echo "Simulation output did not report completed iterations" >&2
    exit 1
}
completed_iterations="${BASH_REMATCH[2]}"
[[ "$summary" =~ (^|[[:space:]])recorded_frames=([0-9]+) ]] || {
    echo "Simulation output did not report a recorded frame count" >&2
    exit 1
}
recorded_frames="${BASH_REMATCH[2]}"

[[ "$completed_iterations" -eq "$ITERS" ]] || {
    echo "Simulation completed $completed_iterations of $ITERS iterations" >&2
    exit 1
}
[[ "$recorded_frames" -eq "$expected_frames" ]] || {
    echo "Recorded $recorded_frames frames; expected $expected_frames" >&2
    exit 1
}
[[ -s "$OUTPUT" ]] || {
    echo "Trajectory was not created or is empty: $OUTPUT" >&2
    exit 1
}

file_size="$(stat -c '%s' "$OUTPUT")"
checksum="$(sha256sum "$OUTPUT")"
checksum="${checksum%% *}"

echo "Completed iterations: $completed_iterations"
echo "Recorded frame count: $recorded_frames"
echo "Resulting file size: $file_size bytes"
echo "SHA-256 checksum: $checksum"
echo "Trajectory path: $OUTPUT"
