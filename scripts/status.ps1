$r = "C:\Users\lbx13\Doubao\chats\2026-11-24\new-chat-7\stp-plus-plus\runs\pretrain"
$dirs = Get-ChildItem $r -Directory | Where-Object { $_.Name -like 'ce25*' }
$have = 0
foreach ($d in $dirs) {
    $rj = Join-Path $d.FullName 'run.json'
    if (Test-Path $rj) {
        $have++
        $j = Get-Content $rj -Raw | ConvertFrom-Json
        "{0,-16} ppl={1:0.0000}" -f $d.Name, $j.eval.ppl
    }
}
"done=$have/9"
"now: $(Get-Date -Format 'HH:mm')"
