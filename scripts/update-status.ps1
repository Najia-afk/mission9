# Mission9 - Auto-update status.json every 10 seconds
# Usage: .\scripts\update-status.ps1

$clusterIdFile = "C:\git\mission9\infra\.cluster-id"
$statusFile = "C:\git\mission9\infra\status.json"

Write-Host "`n  ====== MISSION9 STATUS UPDATER ======" -ForegroundColor Cyan
Write-Host "  Updating status.json every 10 seconds..." -ForegroundColor Gray
Write-Host "  Press Ctrl+C to stop`n" -ForegroundColor Gray

Set-Location "C:\git\mission9"

while ($true) {
    try {
        # Get active cluster (non-terminated)
        $clusters = docker compose run --rm aws emr list-clusters --region eu-west-1 --query "Clusters[?Status.State!='TERMINATED' && Status.State!='TERMINATED_WITH_ERRORS'].Id" --output text 2>$null
        $clusterId = ($clusters -split "`t")[0]
        
        if ($clusterId) {
            # Get cluster details
            $details = docker compose run --rm aws emr describe-cluster --cluster-id $clusterId --region eu-west-1 --query "Cluster.{emr_cluster_id:Id,emr_state:Status.State,emr_master_dns:MasterPublicDnsName,reason:Status.StateChangeReason.Message}" --output json 2>$null | ConvertFrom-Json
            
            # Get S3 bucket
            $bucket = docker compose run --rm aws s3 ls --region eu-west-1 2>$null | Select-String "mission9-data" | ForEach-Object { $_.Line.Split()[-1] } | Select-Object -First 1
            
            # Count files in S3 if bucket exists
            $fileCount = 0
            if ($bucket) {
                $fileCount = (docker compose run --rm aws s3 ls "s3://$bucket/" --recursive 2>$null | Measure-Object).Count
            }
            
            # Build status object
            $status = @{
                emr_cluster_id = $details.emr_cluster_id
                emr_state = $details.emr_state
                emr_master_dns = $details.emr_master_dns
                emr_reason = $details.reason
                s3_bucket = $bucket
                s3_file_count = $fileCount
                region = "eu-west-1"
                project = "mission9"
                timestamp = (Get-Date -Format o)
                jupyterhub_url = if ($details.emr_master_dns) { "https://$($details.emr_master_dns):9443" } else { $null }
            }
            
            # Write JSON
            $status | ConvertTo-Json | Out-File -Encoding utf8 $statusFile
            
            # Console output
            $stateColor = switch ($details.emr_state) {
                "WAITING" { "Green" }
                "RUNNING" { "Green" }
                "STARTING" { "Yellow" }
                "BOOTSTRAPPING" { "Yellow" }
                default { "Red" }
            }
            
            $time = Get-Date -Format "HH:mm:ss"
            Write-Host "  [$time] " -NoNewline
            Write-Host "$($details.emr_state)" -ForegroundColor $stateColor -NoNewline
            Write-Host " | Cluster: $clusterId" -NoNewline
            if ($details.emr_master_dns) {
                Write-Host " | DNS: $($details.emr_master_dns)" -ForegroundColor Cyan
            } else {
                Write-Host " | S3 files: $fileCount"
            }
            
            # Exit if cluster is ready
            if ($details.emr_state -eq "WAITING") {
                Write-Host "`n  ✅ CLUSTER READY!" -ForegroundColor Green
                Write-Host "  JupyterHub: https://$($details.emr_master_dns):9443" -ForegroundColor Cyan
                Write-Host "  Credentials: jovyan / jupyter`n" -ForegroundColor White
            }
        } else {
            $time = Get-Date -Format "HH:mm:ss"
            Write-Host "  [$time] No active cluster found" -ForegroundColor Gray
            
            @{
                emr_cluster_id = $null
                emr_state = "NO_CLUSTER"
                emr_master_dns = $null
                s3_bucket = $null
                region = "eu-west-1"
                project = "mission9"
                timestamp = (Get-Date -Format o)
            } | ConvertTo-Json | Out-File -Encoding utf8 $statusFile
        }
    } catch {
        Write-Host "  [$(Get-Date -Format 'HH:mm:ss')] Error: $_" -ForegroundColor Red
    }
    
    Start-Sleep -Seconds 10
}
