#!/bin/bash
# =====================================================================
# Launch all 5 BNT ADNI training jobs, then summarize results
# =====================================================================
#
# HOW TO USE:
#   bash jobs/run_all_adni.sh
#
# This script submits 5 jobs (one per label) and then submits a
# summary job that waits for all of them to finish.
# =====================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

LABELS="Sex_Binary CDR_Binary degradation_binary_1year degradation_binary_2years degradation_binary_3years"

JOB_IDS=""
for L in $LABELS; do
    JID=$(sbatch --parsable jobs/train_adni.sh $L)
    echo "Submitted $L -> Job $JID"
    if [ -z "$JOB_IDS" ]; then
        JOB_IDS="$JID"
    else
        JOB_IDS="$JOB_IDS:$JID"
    fi
done

# Submit summary job that depends on all training jobs
sbatch --dependency=afterany:$JOB_IDS \
    --job-name=bnt-summary \
    --cpus-per-task=1 \
    --mem=4G \
    --time=00:10:00 \
    --output=logs/summary_%j.out \
    --error=logs/summary_%j.err \
    --wrap="
source /sci/labs/arieljaffe/dan.abergel1/torch_env/bin/activate
cd /sci/labs/arieljaffe/dan.abergel1/repos/BrainNetworkTransformer
python3 scripts/summarize_results.py logs/train_adni_*.out
"

echo ""
echo "All jobs submitted. Summary will run after all training completes."
echo "Check results with: cat logs/summary_*.out"
