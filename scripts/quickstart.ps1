# STP++ quickstart (Windows / PowerShell)
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\quickstart.ps1

$ErrorActionPreference = "Stop"

$ROOT = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$VP   = "F:\aiwork\envs\stp\Scripts\python.exe"
$MODEL = "F:\aiwork\models\Llama-3.2-1B-Instruct"
$DATA  = "F:\aiwork\llm-jepa-ref\datasets"

$env:PYTHONPATH = $ROOT
$env:HF_HOME = "F:\aiwork\hf"
$env:HF_ENDPOINT = "https://hf-mirror.com"

if (-not (Test-Path $VP))    { throw "venv not found: $VP" }
if (-not (Test-Path $MODEL)) { throw "model not found: $MODEL" }

Write-Host "== geometry unit checks ==" -ForegroundColor Cyan
& $VP "$ROOT\tests\test_geometry.py"

Write-Host "`n== 20-step smoke: STP arm ==" -ForegroundColor Cyan
& $VP -m stpp.train --model $MODEL `
    --train_file "$DATA\synth_train.jsonl" `
    --out_dir "$ROOT\runs\quickstart" `
    --epochs 1 --bs 8 --lr 2e-4 --fraction 0.05 --max_steps 20 `
    --geom_kind stp_random_span --lbd 0.02 --log_every 5

Write-Host "`n== eval + probe ==" -ForegroundColor Cyan
& $VP -m stpp.evaluate --model $MODEL --adapter "$ROOT\runs\quickstart\adapter" `
    --test_file "$DATA\synth_test.jsonl" --out "$ROOT\runs\quickstart\eval.json" `
    --limit 40 --dump_traj --limit_traj 40 --anchors_per_seq 12

& $VP -m stpp.eval_tf --model $MODEL --adapter "$ROOT\runs\quickstart\adapter" `
    --test_file "$DATA\synth_test.jsonl" --out "$ROOT\runs\quickstart\tf.json" --limit 100

& $VP -m stpp.probe --traj "$ROOT\runs\quickstart\eval.traj.npz" --out "$ROOT\runs\quickstart\probe.json"

Write-Host "`nDone. See runs\quickstart\" -ForegroundColor Green
