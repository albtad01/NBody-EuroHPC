#!/bin/bash
#SBATCH --account=EUHPC_TDEMO_26
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=nbody_4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=00:20:00
#SBATCH --output=nbody_4gpu_%j.out
#SBATCH --error=nbody_4gpu_%j.err

# First supported gpu+multinode topology: one Booster node, four ranks, four A100s.
set -euo pipefail

module purge
module load profile/base
module load gcc/12.2.0
module load cuda/12.2
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

: "${SLURM_JOB_ID:?This script must be submitted with sbatch}"
: "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unavailable}"

ROOT="$(cd "$SLURM_SUBMIT_DIR" && pwd -P)"
BUILD="$ROOT/build-leonardo-multi"
BIN="$BUILD/bin/murb"

git_root="$(git -C "$ROOT" rev-parse --show-toplevel)"
git_root="$(cd "$git_root" && pwd -P)"
[[ "$git_root" == "$ROOT" ]] || {
    echo "Submit this job from the Git worktree root: $ROOT" >&2
    exit 1
}
[[ -x "$BIN" && -f "$BUILD/murb-build.ready" ]] || {
    echo "Missing prebuilt MPI executable or build stamp: $BUILD" >&2
    echo "Build with the leonardo-multi preset before submitting." >&2
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
unexpected_status=""
while IFS= read -r status_line; do
    case "$status_line" in
        "?? nbody_4gpu_${SLURM_JOB_ID}.out"|"?? nbody_4gpu_${SLURM_JOB_ID}.err") ;;
        *) unexpected_status+="$status_line"$'\n' ;;
    esac
done < <(git -C "$ROOT" status --porcelain --untracked-files=normal)
[[ -z "$unexpected_status" ]] || {
    echo "The worktree is not clean (apart from this job's Slurm logs):" >&2
    printf '%s' "$unexpected_status" >&2
    exit 1
}
[[ "$version" == *" cuda=1 "* && "$version" == *" mpi=1"* ]] || {
    echo "A CUDA- and MPI-enabled executable is required: $version" >&2
    exit 1
}

N="${MURB_N:-10000}"
ITERS="${MURB_ITERS:-20}"
WARMUP="${MURB_WARMUP:-3}"
DT="${MURB_DT:-3600}"
SCHEME="${MURB_SCHEME:-galaxy}"
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
[[ "$SCHEME" == "galaxy" || "$SCHEME" == "random" ]] || {
    echo "MURB_SCHEME must be galaxy or random: $SCHEME" >&2
    exit 1
}
awk -v value="$DT" 'BEGIN {
    valid = value ~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$/ && value + 0 > 0
    exit !valid
}' || {
    echo "MURB_DT must be a finite positive number: $DT" >&2
    exit 1
}

record_args=()
OUTPUT=""
if [[ -n "${MURB_OUTPUT:-}" ]]; then
    [[ "$MURB_OUTPUT" == /* && "$MURB_OUTPUT" == *.murbtraj ]] || {
        echo "MURB_OUTPUT must be an absolute path ending in .murbtraj" >&2
        exit 1
    }
    output_directory="$(dirname -- "$MURB_OUTPUT")"
    output_name="$(basename -- "$MURB_OUTPUT")"
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
    [[ "$RECORD_EVERY" -le "$ITERS" ]] || {
        echo "MURB_RECORD_EVERY must not exceed MURB_ITERS" >&2
        exit 1
    }
    record_args=(--record "$OUTPUT" --record-every "$RECORD_EVERY")
fi

echo "SLURM job ID: $SLURM_JOB_ID"
echo "Git SHA: $revision"
echo "Node: $(hostname)"
echo "MPI rank count: ${SLURM_NTASKS:-4}"
echo "GPU count: 4"
echo "CUDA version:"
nvcc --version
echo "MPI version:"
mpirun --version
echo "Rank-to-GPU mapping:"
srun --nodes=1 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=8 \
    --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 --cpu-bind=cores \
    bash -c 'assigned="${CUDA_VISIBLE_DEVICES:?Slurm GPU binding missing}"; assigned="${assigned%%,*}"; model="$(nvidia-smi --id="$assigned" --query-gpu=name,pci.bus_id --format=csv,noheader)"; printf "rank=%s local_rank=%s CUDA_VISIBLE_DEVICES=%s gpu=%s\n" "$SLURM_PROCID" "$SLURM_LOCALID" "$CUDA_VISIBLE_DEVICES" "$model"'
echo "N: $N"
echo "Iterations: $ITERS"
echo "Warm-up iterations: $WARMUP"
echo "Timestep: $DT"
echo "Scheme: $SCHEME"
if [[ -n "$OUTPUT" ]]; then
    echo "Recording stride: $RECORD_EVERY"
    echo "Expected frame count: $((ITERS / RECORD_EVERY))"
    echo "Output path: $OUTPUT"
else
    echo "Recording: disabled"
fi
echo "$version"

export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OMP_PLACES=cores
export OMP_PROC_BIND=close

run_output="$(srun --nodes=1 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=8 \
    --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 --cpu-bind=cores \
    "$BIN" -n "$N" -i "$ITERS" --warmup "$WARMUP" \
    --im gpu+multinode --scheme "$SCHEME" --nv --gf -v --dt "$DT" \
    "${record_args[@]}")"
printf '%s\n' "$run_output"

summary=""
while IFS= read -r line; do
    [[ "$line" == completed_iterations=* ]] && summary="$line"
done <<< "$run_output"
[[ "$summary" =~ (^|[[:space:]])completed_iterations=([0-9]+) ]] || {
    echo "Multi-GPU run did not report completed iterations" >&2
    exit 1
}
completed_iterations="${BASH_REMATCH[2]}"
[[ "$completed_iterations" -eq "$ITERS" ]] || {
    echo "Multi-GPU run completed $completed_iterations of $ITERS iterations" >&2
    exit 1
}
[[ "$summary" =~ (^|[[:space:]])compute_ms=([^[:space:]]+) ]] || {
    echo "Multi-GPU run did not report compute timing" >&2
    exit 1
}
compute_ms="${BASH_REMATCH[2]}"
[[ "$summary" =~ (^|[[:space:]])interactions_per_second=([^[:space:]]+) ]] || {
    echo "Multi-GPU run did not report interactions/s" >&2
    exit 1
}
interactions_per_second="${BASH_REMATCH[2]}"
[[ "$summary" =~ (^|[[:space:]])estimated_GFLOP_per_second=([^[:space:]]+) ]] || {
    echo "Multi-GPU run did not report estimated GFLOP/s" >&2
    exit 1
}
estimated_gflops="${BASH_REMATCH[2]}"

echo "Completed iterations: $completed_iterations"
echo "Compute timing: $compute_ms ms"
echo "Interactions/s: $interactions_per_second"
echo "Estimated GFLOP/s: $estimated_gflops"
if [[ -n "$OUTPUT" ]]; then
    expected_frames=$((ITERS / RECORD_EVERY))
    [[ "$summary" =~ (^|[[:space:]])recorded_frames=([0-9]+) ]] || {
        echo "Multi-GPU run did not report a recorded frame count" >&2
        exit 1
    }
    recorded_frames="${BASH_REMATCH[2]}"
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
    echo "Recorded frame count: $recorded_frames"
    echo "Resulting file size: $file_size bytes"
    echo "SHA-256 checksum: $checksum"
    echo "Trajectory path: $OUTPUT"
fi
