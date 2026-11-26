import pathlib, zipfile, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT.parent / "stp-plus-plus.zip"

code_globs = ["*.py", "*.md", "*.ps1", "*.txt", "*.toml", "*.cfg"]
result_globs = ["results.json", "run.json", "tf.json", "eval.json", "probe.json", "PILOT.md", "*.results.json"]

files = []
for g in code_globs:
    files += [p for p in ROOT.rglob(g) if "runs" not in p.parts and "__pycache__" not in p.parts]
for g in result_globs:
    files += [p for p in (ROOT / "runs").rglob(g)]

files = sorted(set(files))
if OUT.exists():
    OUT.unlink()
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p, p.relative_to(ROOT.parent))

print("files:", len(files))
print("size MB: %.3f" % (OUT.stat().st_size / 1e6))
for p in files:
    if p.suffix in {".md", ".py"} and "runs" not in p.parts:
        pass
print("new in root:", sorted(p.name for p in files if p.parent == ROOT))
