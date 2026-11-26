$root = $PSScriptRoot | Split-Path -Parent          # repo root (parent of scripts/)
$r = Join-Path $root "runs\pretrain"
$dirs = Get-ChildItem $r -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -notmatch 'smoke' }
$have = 0
foreach ($d in $dirs) {
    $rj = Join-Path $d.FullName 'run.json'
    if (Test-Path $rj) {
        $have++
        $j = Get-Content $rj -Raw | ConvertFrom-Json
        "{0,-16} ppl={1:0.0000}" -f $d.Name, $j.eval.ppl
    }
}
"completed=$have"
"now: $(Get-Date -Format 'HH:mm')"
