import json
import run_benchmark_eval

with open('benchmark_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

single_cor = 0
jury_cor = 0

for c in data['cases']:
    cid = c['case_id']
    sp, sr = run_benchmark_eval.grade_case(cid, c['single_baseline']['text'])
    c['single_baseline']['passed'] = sp
    c['single_baseline']['rationale'] = sr
    
    jp, jr = run_benchmark_eval.grade_case(cid, c['jury_deliberation']['final_verdict'])
    c['jury_deliberation']['passed'] = jp
    c['jury_deliberation']['rationale'] = jr
    
    single_cor += int(sp)
    jury_cor += int(jp)
    print(f"[{cid}] Single: {sp} ({sr}) | Jury: {jp} ({jr})")

total = len(data['cases'])
data['metrics']['single_correct_count'] = single_cor
data['metrics']['jury_correct_count'] = jury_cor
data['metrics']['single_model_accuracy_pct'] = round(single_cor / total * 100, 1)
data['metrics']['jury_accuracy_pct'] = round(jury_cor / total * 100, 1)
data['metrics']['improvement_pct'] = round((jury_cor - single_cor) / total * 100, 1)
data['metrics']['both_correct_count'] = sum(
    1 for c in data['cases'] if c['single_baseline']['passed'] and c['jury_deliberation']['passed']
)
data['metrics']['both_wrong_count'] = sum(
    1 for c in data['cases'] if not c['single_baseline']['passed'] and not c['jury_deliberation']['passed']
)
data['metrics']['errors_corrected_by_jury'] = sum(
    1 for c in data['cases'] if not c['single_baseline']['passed'] and c['jury_deliberation']['passed']
)
data['metrics']['wrong_corrections_by_jury'] = sum(
    1 for c in data['cases'] if c['single_baseline']['passed'] and not c['jury_deliberation']['passed']
)

with open('benchmark_results.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print("\n--- RECALCULATED METRICS ---")
print(json.dumps(data['metrics'], indent=2))
