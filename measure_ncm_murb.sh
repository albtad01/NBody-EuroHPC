#!/usr/bin/env bash
# measure_ncm_murb.sh
# Submit a Slurm job on az4-n4090 (node az4-n4090-1) that runs ./bin/murb
# while continuously collecting NCM power samples (node-conso -t 5).
# After the job finishes the script computes mean/variance power per component
# (GPU, CPU, RAM, MOTHERBOARD) and writes a human-readable report.
#
# Usage:
#   ./measure_ncm_murb.sh [-n N] [-i I] [--im IM] [--help]
# Examples:
#   ./measure_ncm_murb.sh -n 500000 -i 100 --im gpu+tile

set -euo pipefail

# Defaults
N_VAL=500000
I_VAL=100
IM_VAL="gpu+tile"
PARTITION="az4-n4090"
NODE="az4-n4090-1"
SBATCH_OPTS=("--nodes=1" "--ntasks=1")

show_help(){
  cat <<EOF
Usage: $0 [-n N] [-i I] [--im IM] [--help]

Options:
  -n N         -> passes -n N to ./bin/murb (default: ${N_VAL})
  -i I         -> passes -i I to ./bin/murb (default: ${I_VAL})
  --im IM      -> passes --im IM to ./bin/murb (default: ${IM_VAL})
  --help       -> show this help and exit

This script:
  1) generates a Slurm batch script that runs on ${NODE} in partition ${PARTITION}
  2) inside the job: loads ncm module, enables I2C chains, starts measurements,
     runs ./bin/murb in background and repeatedly calls `node-conso -t 5` appending
     to an energy log until murb finishes
  3) when job completes, the script computes mean/variance power per component
     and writes energy_report.<JOBID>.txt

Requirements & assumptions:
  * You have sbatch/squeue available and a shared filesystem (job writes logs
    in the working directory).
  * node-conso is available via module ncm/gdcc873f on the compute node.
  * The murb binary exists at ./bin/murb relative to the job working directory.
  * This script submits the job with --nodelist=${NODE} to force the node.

EOF
}

# parse args (supporting --im long option)
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n)
      N_VAL="$2"; shift 2;;
    -i)
      I_VAL="$2"; shift 2;;
    --im)
      IM_VAL="$2"; shift 2;;
    --help|-h)
      show_help; exit 0;;
    *)
      echo "Unknown option: $1" >&2; show_help; exit 1;;
  esac
done

# Create output directory based on --im value (replace + with _)
IM_DIR="power_tracking_${IM_VAL//+/_}"
mkdir -p "$IM_DIR"

# Create a temporary Slurm batch script
BATCH_FILE="slurm_murb_$$.sh"
cat > "$BATCH_FILE" <<'BATCH_EOF'
#!/usr/bin/env bash
#SBATCH --job-name=murb_ncm
#SBATCH --partition=az4-n4090
#SBATCH --nodelist=az4-n4090-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=murb_job.%j.out
#SBATCH --error=murb_job.%j.err

# job body starts here
set -euo pipefail

# Load NCM module (required to have node-conso in PATH)
module load ncm/gdcc873f || true

# Ensure I2C chains powered and measurements started
node-conso -P 1 || true
node-conso -P 2 || true
node-conso -M 1 || true
node-conso -M 2 || true

ENERGY_LOG="energy.log"
MURB_LOG="murb.log"

# write header with wallclock times (to compute accurate duration later)
echo "# NCM LOG" > "$ENERGY_LOG"
echo "# HOST: $(hostname)" >> "$ENERGY_LOG"
echo "# START_WALLCLOCK: $(date +%s.%N)" >> "$ENERGY_LOG"

# start murb in background
./bin/murb -n __NVAL__ -i __IVAL__ --im __IMVAL__ --nv > "$MURB_LOG" 2>&1 &
MURB_PID=$!

# let murb start and reach steady state
node-conso -t 5 > \dev\null

# take exactly one 5-second power measurement
node-conso -t 5 > "$ENERGY_LOG" 2>&1 || true


# final touch: stop measurements and write end time
node-conso -m 1 || true
node-conso -m 2 || true

echo "# END_WALLCLOCK: $(date +%s.%N)" >> "$ENERGY_LOG"

# exit with the same code as murb (wait and capture)
wait "$MURB_PID" || true

BATCH_EOF

# replace placeholders with actual values
sed -i "s|__NVAL__|${N_VAL}|g" "$BATCH_FILE"
sed -i "s|__IVAL__|${I_VAL}|g" "$BATCH_FILE"
sed -i "s|__IMVAL__|${IM_VAL}|g" "$BATCH_FILE"

# submit the job
SUBMIT_OUT=$(sbatch "$BATCH_FILE")
# example: Submitted batch job 12345
JOBID=$(echo "$SUBMIT_OUT" | awk '{print $NF}')

echo "Job submitted (job id: ${JOBID})"

echo "Waiting for the job to start..."
STATE=""
LAST_STATE=""
while true; do
  # query job state; if squeue empty then job finished (or use sacct)
  SOUT=$(squeue -j ${JOBID} -h -o "%T" 2>/dev/null || true)
  if [[ -z "${SOUT}" ]]; then
    # job is no longer in squeue: finished (could be COMPLETED, FAILED, CANCELLED)
    echo "Job no longer in squeue. Waiting for finalization..."
    break
  fi
  STATE=$(echo "$SOUT" | tr -d '\n' | tr -s ' ')
  if [[ "$STATE" != "$LAST_STATE" ]]; then
    if [[ "$STATE" == "PENDING" ]]; then
      echo "Job pending"
    elif [[ "$STATE" == "CONFIGURING" ]] || [[ "$STATE" == "COMPLETING" ]] ; then
      echo "Job ${STATE}"
    elif [[ "$STATE" == "RUNNING" ]]; then
      echo "Job running"
    else
      echo "Job state: ${STATE}"
    fi
    LAST_STATE="$STATE"
  fi
  sleep 5
done

# final: wait for job output files to appear and for sacct to have info
# fetch job exit status via sacct if available
if command -v sacct >/dev/null 2>&1; then
  # wait a bit for sacct records
  sleep 2
  JOB_EXIT=$(sacct -j ${JOBID} -n -o State%20 -P | head -n1 | cut -d'|' -f1 || true)
  echo "Job completed (sacct state: ${JOB_EXIT})"
else
  echo "Job completed (sacct not available)
"
fi

# At this point energy.<JOBID>.log should exist in the working dir
mv energy.log "${IM_DIR}"
mv murb.log "${IM_DIR}"
mv murb_job* "${IM_DIR}"
ENERGY_FILE="${IM_DIR}/energy.log"
REPORT_FILE="${IM_DIR}/energy_report.txt"

if [[ ! -f "$ENERGY_FILE" ]]; then
  echo "ERROR: energy file $ENERGY_FILE not found in current directory." >&2
  echo "The job probably didn't run on a shared filesystem or failed before creating the log." >&2
  exit 1
fi

cd "${IM_DIR}"

# Analyze the energy file using an embedded python script
python3 - <<'PY'
import re, sys, statistics
from collections import defaultdict, OrderedDict

energy_file = sys.argv[1] if len(sys.argv)>1 else 'energy.log'
report_file = sys.argv[2] if len(sys.argv)>2 else 'energy_report.txt'

# parse header for wallclock times
start_wall=None; end_wall=None
samples_by_ts = OrderedDict()  # ts -> { (probe,chan): power }

pat = re.compile(r'^\s*(\d+)\s+([0-9]+\.[0-9]+)\s+\S+\s+([0-9.]+)\s*V\s+([\-0-9.]+)\s*A')
# alternative pattern if columns spaced differently
pat2 = re.compile(r'^\s*(\d+)\s+([0-9]+\.[0-9]+)\s+\S+\s*([0-9.]+)\s*V\s*([\-0-9.]+)\s*A')

with open(energy_file) as f:
    for line in f:
        line=line.rstrip('\n')
        if line.startswith('# START_WALLCLOCK:'):
            start_wall=float(line.split(':',1)[1].strip())
            continue
        if line.startswith('# END_WALLCLOCK:'):
            end_wall=float(line.split(':',1)[1].strip())
            continue
        m = pat.match(line)
        if not m:
            m = pat2.match(line)
        if not m:
            continue
        ts = int(m.group(1))
        probe_chan = m.group(2)  # like '0.4'
        volts = float(m.group(3))
        amps = float(m.group(4))
        power = volts * amps
        if ts not in samples_by_ts:
            samples_by_ts[ts] = {}
        samples_by_ts[ts][probe_chan] = power

if not samples_by_ts:
    print('No samples parsed from', energy_file)
    sys.exit(2)

# map groups
GPU_keys = ['1.0','1.1','1.2','1.3']
CPU_key = '0.4'
RAM_key = '0.2'
MB_key = '0.3'

gpu_vals=[]; cpu_vals=[]; ram_vals=[]; mb_vals=[]; total_vals=[]
for ts, d in samples_by_ts.items():
    gpu = sum(d.get(k,0.0) for k in GPU_keys)
    cpu = d.get(CPU_key, 0.0)
    ram = d.get(RAM_key, 0.0)
    mb = d.get(MB_key, 0.0)
    total = sum(d.values())
    gpu_vals.append(gpu)
    cpu_vals.append(cpu)
    ram_vals.append(ram)
    mb_vals.append(mb)
    total_vals.append(total)

n = len(total_vals)
avg = lambda arr: statistics.mean(arr) if arr else 0.0
pvar = lambda arr: statistics.pvariance(arr) if len(arr)>1 else 0.0
pmin = lambda arr: min(arr) if arr else 0.0
pmax = lambda arr: max(arr) if arr else 0.0

# compute duration
if start_wall is not None and end_wall is not None:
    total_time = end_wall - start_wall
else:
    # fallback: estimate time by assuming uniform spacing and 5s per node-conso call
    # use count of samples and assume approx 5s per chunk is unreliable; set total_time = n* (5.0 / max(1,n))
    total_time = float(n) * 0.001  # fallback tiny number to avoid div0

# energy per timestamp (using equal dt = total_time / n)
dt = total_time / n if n>0 else 0.0
energy_per_ts = [p * dt for p in total_vals]

report_lines = []
report_lines.append('SUMMARY (job: {} )'.format(sys.argv[1] if len(sys.argv)>1 else ''))
report_lines.append('Total samples (timestamps): {}'.format(n))
report_lines.append('Measured wallclock duration: {:.3f} seconds'.format(total_time))
report_lines.append('')
report_lines.append('MEAN WATTAGE (W):')
report_lines.append('  GPU   : {:.3f} W'.format(avg(gpu_vals)))
report_lines.append('  CPU   : {:.3f} W'.format(avg(cpu_vals)))
report_lines.append('  RAM   : {:.3f} W'.format(avg(ram_vals)))
report_lines.append('  MBOARD: {:.3f} W'.format(avg(mb_vals)))
report_lines.append('  TOTAL : {:.3f} W'.format(avg(total_vals)))
report_lines.append('')
report_lines.append('POWER VARIANCE (W^2):')
report_lines.append('  GPU   : {:.6f}'.format(pvar(gpu_vals)))
report_lines.append('  CPU   : {:.6f}'.format(pvar(cpu_vals)))
report_lines.append('  RAM   : {:.6f}'.format(pvar(ram_vals)))
report_lines.append('  MBOARD: {:.6f}'.format(pvar(mb_vals)))
report_lines.append('  TOTAL : {:.6f}'.format(pvar(total_vals)))
report_lines.append('')
report_lines.append('ENERGY (estimated) over {:.3f} s:'.format(total_time))
report_lines.append('  TOTAL ENERGY mean per-timestamp: {:.6f} J (dt={:.6f}s)'.format(statistics.mean(energy_per_ts) if energy_per_ts else 0.0, dt))
report_lines.append('  TOTAL ENERGY sum (estimate): {:.6f} J'.format(sum(energy_per_ts)))
report_lines.append('  ENERGY variance (per-timestamp) : {:.6f} (J^2)'.format(statistics.pvariance(energy_per_ts) if len(energy_per_ts)>1 else 0.0))
report_lines.append('')
report_lines.append('ADDITIONAL METRICS:')
report_lines.append('  Samples per second (approx): {:.3f}'.format(n/total_time if total_time>0 else 0.0))
report_lines.append('  TOTAL POWER min/max: {:.3f}/{:.3f} W'.format(pmin(total_vals), pmax(total_vals)))

with open(report_file, 'w') as out:
    out.write('\n'.join(report_lines))

print('\n'.join(report_lines))
print('\nReport written to', report_file)
PY

echo "Analysis complete. Report saved to ${REPORT_FILE}"

echo "Done."

# cleanup
rm -f "$BATCH_FILE"

exit 0

