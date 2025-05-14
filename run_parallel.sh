#!/bin/bash

# Define your experiments
exps=(
#    "topk_context"
#    "targeted_attacks_oneQ_oneA"
    "targeted_attacks_multiQ_oneA"
    "targeted_attacks_multiQ_multiA"
#    "mask_attack"
    "judge_defence"
#    "copali_ab"
#    "copali_ab_cpoiT"
#    "perturbation_plot"
#    "perturbation_plot_targeted"
#    "transferability"
)

# Define a function to run one experiment
run_exp() {
    exp="$1"
#    echo "=== [1/2] Training: $exp ==="
#    python src/attack_train.py --config-name "$exp"
#    if [ $? -ne 0 ]; then
#        echo "❌ Training failed for $exp. Skipping eval."
#        return
#    fi

    echo "=== [2/2] Evaluating: $exp ==="
    python src/attack_eval.py --config-name "$exp"
    if [ $? -ne 0 ]; then
        echo "❌ Evaluation failed for $exp."
    else
        echo "✅ Done with $exp"
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

echo "🎉 All experiments processed."