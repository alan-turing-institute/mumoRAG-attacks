#!/bin/bash

# Define your experiments
exps=(
    "topk_context"
    "targeted_attacks_oneQ_oneA"
    "targeted_attacks_multiQ_oneA"
    "targeted_attacks_multiQ_multiA"
    "mask_attack"
    "judge_defence"
    "copali_ab"
    "copali_ab_cpoiT"
    "perturbation_plot"
    "perturbation_plot_targeted"
    "transferability"
)

# Loop through each config
for exp in "${exps[@]}"; do
    echo "=== [1/2] Training: $exp ==="
    python src/attack_train.py --config-name "$exp"
    if [ $? -ne 0 ]; then
        echo "❌ Training failed for $exp. Skipping eval."
        continue
    fi

    echo "=== [2/2] Evaluating: $exp ==="
    python src/attack_eval.py --config-name "$exp"
    if [ $? -ne 0 ]; then
        echo "❌ Evaluation failed for $exp."
    else
        echo "✅ Done with $exp"
    fi
done

echo "🎉 All experiments processed."
