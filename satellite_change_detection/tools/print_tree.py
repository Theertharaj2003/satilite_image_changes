from pathlib import Path
root=Path(__file__).resolve().parents[1]
for p in sorted(root.rglob('*')):
    if any(part.startswith('.') for part in p.parts):
        continue
    rel=p.relative_to(root)
    if len(rel.parts)<=3:
        print(rel)
