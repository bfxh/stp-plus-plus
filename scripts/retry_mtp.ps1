$root = "C:\Users\lbx13\Doubao\chats\2026-11-24\new-chat-7\stp-plus-plus"
$py = "F:\aiwork\envs\stp\Scripts\python.exe"
$arms = @("ce33", "ce33_mtp2", "mtp2", "mtp3", "ce25_mtp3", "mtp3_l", "ce25_mtp3_l")
for ($pass = 1; $pass -le 5; $pass++) {
    "=== pass $pass  $(Get-Date -Format 'HH:mm') ==="
    $a = @("$root\scripts\run_pretrain.py", "--arms") + $arms + @("--seeds", "82", "23", "37", "--steps", "120", "--bs", "4", "--out-root", "runs\pretrain")
    & $py @a 2>&1 | Select-String "ppl=|fail|pass"
    Start-Sleep -Seconds 45
}
"done $(Get-Date -Format 'HH:mm')"
