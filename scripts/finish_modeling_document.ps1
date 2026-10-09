$ErrorActionPreference = 'Stop'
$projectRoot = 'C:\Users\id116\Desktop\komap'
$documentWork = Join-Path $projectRoot 'reports\modeling_document_work'
$pythonRuntime = 'C:\Users\id116\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$sshKey = Join-Path $env:USERPROFILE '.ssh\id_ed25519'
$serverName = 'root@202.181.159.219'
$env:PYTHONIOENCODING = 'utf-8'
$env:PATH = 'C:\Users\id116\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin;' + $env:PATH
Set-Location -LiteralPath $projectRoot
function Save-DocumentStatus($value) {
    $value.time = [DateTimeOffset]::Now.ToString('o')
    $value | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $documentWork 'document_state.json') -Encoding UTF8
}
try {
    Save-DocumentStatus @{stage='waiting_for_seed0_results';pid=$PID;final_document='reports/모델링결과서_김형수없는김형수.docx'}
    $deadline = [DateTimeOffset]::Now.AddHours(8)
    $completed = $false
    while ([DateTimeOffset]::Now -lt $deadline) {
        $probe = @'
import pathlib,json
p=pathlib.Path('/workspace/komap/runs/review16_additional')
done=p/'all.done';error=p/'error.json'
if done.exists(): print(json.dumps({'stage':'done','receipt':json.loads(done.read_text())}))
elif error.exists(): print(json.dumps({'stage':'error','error':json.loads(error.read_text())}))
else: print(json.dumps({'stage':'waiting','state':json.loads((p/'state.json').read_text())}))
'@
        $response = $probe | ssh -o BatchMode=yes -o ConnectTimeout=15 -p 13980 -i $sshKey $serverName python3 -
        if ($LASTEXITCODE -eq 0) {
            $remoteState = $response | ConvertFrom-Json
            if ($remoteState.stage -eq 'error') { throw ($remoteState.error | ConvertTo-Json -Compress) }
            if ($remoteState.stage -eq 'done') {
                if ($remoteState.receipt.outcome -ne 'three_seed0_jobs_full_cv_and_final_report_complete') { throw 'Unexpected completion scope' }
                $completed = $true
                break
            }
            Save-DocumentStatus @{stage='waiting_for_seed0_results';pid=$PID;remote=$remoteState.state}
        }
        Start-Sleep -Seconds 60
    }
    if (-not $completed) { throw 'Timed out waiting for seed0 completion; no report with incomplete results created' }
    $resultDirectory = Join-Path $projectRoot 'runs\review16_additional'
    New-Item -ItemType Directory -Path $resultDirectory -Force | Out-Null
    foreach ($name in @('seed0_full_results.json','summary.md','all.done','state.json','completed_jobs.json','seed0_artifacts.json','plan_seed0_closeout.json','seed0_closeout_code_hashes.json')) {
        scp -q -o BatchMode=yes -o ConnectTimeout=15 -P 13980 -i $sshKey "${serverName}:/workspace/komap/runs/review16_additional/$name" (Join-Path $resultDirectory $name)
        if ($LASTEXITCODE -ne 0) { throw "Failed to retrieve $name" }
    }
    scp -q -o BatchMode=yes -o ConnectTimeout=15 -P 13980 -i $sshKey "${serverName}:/workspace/komap/reports/round16_seed0_final.md" (Join-Path $projectRoot 'reports\round16_seed0_final.md')
    if ($LASTEXITCODE -ne 0) { throw 'Failed to retrieve report evidence' }
    Save-DocumentStatus @{stage='building_docx';pid=$PID}
    & $pythonRuntime scripts/build_modeling_report.py
    if ($LASTEXITCODE -ne 0) { throw 'Final DOCX builder failed' }
    $documentPath = Join-Path $projectRoot 'reports\모델링결과서_김형수없는김형수.docx'
    $finalPdf = Join-Path $documentWork 'final_native_word.pdf'
    $wordApp=$null; $finalDoc=$null
    try {
        $wordApp=New-Object -ComObject Word.Application
        $wordApp.Visible=$false; $wordApp.DisplayAlerts=0; $wordApp.AutomationSecurity=3
        $finalDoc=$wordApp.Documents.Open($documentPath,$false,$true,$false)
        $pageCount=$finalDoc.ComputeStatistics(2)
        if ($pageCount -gt 10) { throw "Report exceeds conservative ten-page total limit: $pageCount" }
        $finalDoc.ExportAsFixedFormat($finalPdf,17)
    } finally {
        if ($finalDoc) { $finalDoc.Close(0) }
        if ($wordApp) { $wordApp.Quit(); [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($wordApp) }
    }
    & $pythonRuntime reports/modeling_document_work/render_with_word_pdf.py $documentPath $finalPdf (Join-Path $documentWork 'final_render')
    if ($LASTEXITCODE -ne 0) { throw 'Final render failed' }
    $originalHash=(Get-FileHash -LiteralPath 'C:\Users\id116\Downloads\모델링결과서 (2).doc' -Algorithm SHA256).Hash
    $retainedHash=(Get-FileHash -LiteralPath (Join-Path $documentWork 'reference_original.doc') -Algorithm SHA256).Hash
    if ($originalHash -ne $retainedHash) { throw 'Reference changed; fidelity must be checked again' }
    Save-DocumentStatus @{stage='rendered_awaiting_visual_review';pid=$PID;pages=$pageCount;document=$documentPath;training_complete=$true;further_training_scheduled=$false}
} catch {
    Save-DocumentStatus @{stage='error';pid=$PID;error=$_.Exception.Message}
    throw
}
