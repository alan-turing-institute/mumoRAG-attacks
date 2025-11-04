#!/bin/bash

# Define your experiments
exps=(
# "rebuttal_non_targeted"
# "rebuttal_targeted_attacks_oneQ_oneA"

# --------

#  "paper_non_targeted"
#  "paper_targeted_attacks_oneQ_oneA"
#  "paper_targeted_attacks_multiQ_oneA"
#  "paper_targeted_attacks_multiQ_multiA"
#  "paper_multi_transferability"
#  "paper_multi_transferability_targeted"
#  "paper_judge_defence"
#  "paper_judge_defence_adapt"
#  "paper_judge_defence_targeted"
#  "paper_judge_defence_targeted_adapt"

#  "paper_topk_context_targeted"
#  "paper_topk_context"

#  "paper_leave_one_out_multi_transferability"
#  "paper_leave_one_out_multi_transferability_targeted"
#
# "paper_defences"
# "paper_targeted_defences"
# "paper_non_targeted"
# "paper_targeted_attacks_oneQ_oneA"

#  "paper_GPT_non_targeted"
#  "paper_GPT_targeted_attacks_oneQ_oneA"
#  "paper_GPT_targeted_attacks_multiQ_oneA"
#  "paper_GPT_targeted_attacks_multiQ_multiA"

# -------

#"paper_universal_GT_baseline"
#"paper_targeted_1-1_neg_baseline"
#"paper_targeted_m-1_neg_baseline"
#"paper_targeted_m-m_neg_baseline"
#"paper_multi_transferability_GT_baseline"
#"paper_leave_one_out_multi_transferability_GT_baseline"
#"paper_multi_transferability_targeted_neg_baseline"
"paper_leave_one_out_multi_transferability_targeted_neg_baseline"

# -------

  #  "paper_combined_defence"
#  "paper_combined_defence_adapt"
#  "paper_combined_defence_targeted"
#  "paper_combined_defence_targeted_adapt"
#   "paper_defences"
#   "paper_targeted_defences"
#  "paper_perturbation_plot"
#   "paper_copali_ab"
#   "paper_copali_ab_cpoiT"
#    "paper_GPT_non_targeted"
#    "paper_GPT_targeted_attacks_oneQ_oneA"
#    "paper_GPT_targeted_attacks_multiQ_oneA"
#    "paper_GPT_targeted_attacks_multiQ_multiA"
)

declare -a succeeded=()
declare -a failed=()

# Define a function to run one experiment
run_exp() {
    exp="$1"
#    echo "=== [1/2] Training: $exp ==="
#    python src/attack_train.py --config-name "$exp"
#    if [ $? -ne 0 ]; then
#        echo "❌ Training failed for $exp. Skipping eval."
#        failed+=("$exp (train failed)")
#        return
#    fi

    echo "=== [2/2] Evaluating: $exp ==="
    python src/attack_eval.py --config-name "$exp"
    if [ $? -ne 0 ]; then
        echo "❌ Evaluation failed for $exp."
        failed+=("$exp (eval failed)")
    else
        echo "✅ Done with $exp"
        succeeded+=("$exp")
    fi
}

# Run 3 experiments in parallel
parallel=1
count=0

for exp in "${exps[@]}"; do
    run_exp "$exp" &
    ((count++))

    if (( count % parallel == 0 )); then
        wait  # Wait for the current batch to finish
    fi
done

wait  # Wait for any remaining background processes

echo ""
echo "==============================================="
echo "🎯 EXPERIMENT SUMMARY"
echo "==============================================="

if [ ${#succeeded[@]} -gt 0 ]; then
    echo "✅ Succeeded:"
    for exp in "${succeeded[@]}"; do
        echo "  - $exp"
    done
else
    echo "✅ Succeeded: None"
fi

echo ""

if [ ${#failed[@]} -gt 0 ]; then
    echo "❌ Failed:"
    for exp in "${failed[@]}"; do
        echo "  - $exp"
    done
else
    echo "❌ Failed: None"
fi

echo "==============================================="
echo "🎉 All experiments processed."
