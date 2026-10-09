param([switch]$Once)
$ErrorActionPreference = 'Stop'
$projectRoot = 'C:\Users\id116\Desktop\komap'
$monitorDirectory = Join-Path $projectRoot 'runs\review16_additional\hourly_checks'
$documentStatePath = Join-Path $projectRoot 'reports\modeling_document_work\document_state.json'
$sshKey = Join-Path $env:USERPROFILE '.ssh\id_ed25519'
New-Item -ItemType Directory -Path $monitorDirectory -Force | Out-Null
$startedAt = [DateTimeOffset]::Now
$nextCheck = $startedAt.AddHours(1)
$deadline = $startedAt.AddHours(8)
$metadata = @{pid=$PID;started=$startedAt.ToString('o');interval_minutes=60;next_check=$nextCheck.ToString('o');notification_channel='local project records; no scheduled chat messages'}
if (-not $Once) {
    $metadata | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $monitorDirectory 'schedule.json') -Encoding UTF8
}
do {
    if (-not $Once) {
        $waitSeconds=[Math]::Max(0,[Math]::Ceiling(($nextCheck-[DateTimeOffset]::Now).TotalSeconds))
        if ($waitSeconds -gt 0) { Start-Sleep -Seconds $waitSeconds }
    }
    $snapshot=@{checked_at=[DateTimeOffset]::Now.ToString('o');interval_minutes=60}
    try {
        $probe=@'
import pathlib,json,datetime,subprocess
r=pathlib.Path('/workspace/komap');p=r/'runs/review16_additional'
out={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':json.loads((p/'state.json').read_text()),'folds':[]}
for fold in [2,3,4]:
 f=r/f'runs/cv/x4_mit_bw_r90/fold{fold}/log.jsonl';rows=[]
 if f.exists():
  for line in f.read_text().splitlines():
   try:rows.append(json.loads(line))
   except json.JSONDecodeError:pass
 out['folds'].append({'fold':fold,'seed':0,'completed_epochs':rows[-1]['epoch'] if rows else 0,'epochs_total':60,'result_exists':(f.parent/'result.json').exists(),'last_loss':rows[-1]['loss'] if rows else None,'recent_train_sec':sum(x['train_sec'] for x in rows[-5:])/len(rows[-5:]) if rows else None})
for name in ['completed_jobs.json','all.done','error.json']:
 f=p/name
 if f.exists():out[name]=json.loads(f.read_text())
pid=out['state'].get('pid');out['controller_alive']=bool(pid and pathlib.Path(f'/proc/{pid}').exists())
q=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True)
out['gpu']=q.stdout.strip() if q.returncode==0 else 'GPU query failed'
print(json.dumps(out,allow_nan=False))
'@
        $response=$probe | ssh -o BatchMode=yes -o ConnectTimeout=15 -p 13980 -i $sshKey root@202.181.159.219 python3 -
        if ($LASTEXITCODE -ne 0) { throw 'SSH status check failed' }
        $snapshot.server=$response | ConvertFrom-Json
        if (Test-Path -LiteralPath $documentStatePath) {
            $snapshot.document=Get-Content -LiteralPath $documentStatePath -Raw -Encoding UTF8 | ConvertFrom-Json
        }
        $snapshot.check_succeeded=$true
    } catch {
        $snapshot.check_succeeded=$false
        $snapshot.check_error=$_.Exception.Message
    }
    $timestamp=[DateTimeOffset]::Now.ToString('yyyyMMdd_HHmmss')
    $snapshot | ConvertTo-Json -Depth 15 | Set-Content -LiteralPath (Join-Path $monitorDirectory "$timestamp.json") -Encoding UTF8
    $snapshot | ConvertTo-Json -Depth 15 | Set-Content -LiteralPath (Join-Path $monitorDirectory 'latest.json') -Encoding UTF8
    $snapshot | ConvertTo-Json -Depth 15 -Compress | Add-Content -LiteralPath (Join-Path $monitorDirectory 'history.jsonl') -Encoding UTF8
    if ($Once) { $snapshot | ConvertTo-Json -Depth 15; break }
    $terminal=$snapshot.check_succeeded -and (
        $snapshot.server.'error.json' -or $snapshot.document.stage -eq 'error' -or
        ($snapshot.server.'all.done' -and $snapshot.document.stage -eq 'rendered_awaiting_visual_review'))
    if ($terminal) {
        @{stopped_at=[DateTimeOffset]::Now.ToString('o');reason='completed or error';latest='latest.json'} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $monitorDirectory 'stopped.json') -Encoding UTF8
        break
    }
    $nextCheck=$nextCheck.AddHours(1)
    $metadata.next_check=$nextCheck.ToString('o')
    $metadata | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $monitorDirectory 'schedule.json') -Encoding UTF8
} while ([DateTimeOffset]::Now -lt $deadline)
