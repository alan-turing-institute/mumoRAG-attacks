#!/bin/bash

# Define your experiments
exps=(
    "order_in_topk"
    "targeted_attacks"
    "judge_defence"
    "topk_context"
    "copali_ab"
    "perturbation_plot"
    "heatmap_plot"
    "transferability"
    "display_images"
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
