
for i in {0..38}
do
    python scripts/eval/eval_planner_summary.py --eval-path models/eval/planning/object_arrangement/generated_ablation/generated_ablation_trial_$i || true
done

for i in {0..12}
do
    python scripts/eval/eval_planner_summary.py --eval-path models/eval/planning/object_arrangement/oracle/oracle_task_$i || true
done

for i in {0..12}
do
    python scripts/eval/eval_planner_summary.py --eval-path models/eval/planning/object_arrangement/baseline/ablation_task_$i || true
done