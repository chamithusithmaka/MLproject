import json
with open(r'd:\Y4S1\ML\Assignment\MLproject\notebooks\logistic_regression_model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
with open(r'd:\Y4S1\ML\Assignment\MLproject\nb_content.txt', 'w', encoding='utf-8') as out:
    for i, c in enumerate(nb['cells']):
        out.write(f"\n--- Cell {i} ({c['cell_type']}) ---\n")
        out.write(''.join(c['source']) + '\n')
print("Done")
