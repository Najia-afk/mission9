# Mission9 - Live Monitoring Script
# Usage: .\scripts\monitor.ps1

$ErrorActionPreference = "SilentlyContinue"
$refreshInterval = 10

function Get-EMRStatus {
    $cluster = docker compose run --rm aws emr list-clusters --region eu-west-1 --query "Clusters[?Status.State!='TERMINATED'][0]" --output json 2>$null | ConvertFrom-Json
    if ($cluster) {
        $details = docker compose run --rm aws emr describe-cluster --cluster-id $cluster.Id --region eu-west-1 --query "Cluster.{Id:Id,Name:Name,State:Status.State,Reason:Status.StateChangeReason.Message,MasterDns:MasterPublicDnsName}" --output json 2>$null | ConvertFrom-Json
        return $details
    }
    return $null
}

function Get-S3Buckets {
    $buckets = docker compose run --rm aws s3 ls --region eu-west-1 2>$null | Select-String "mission9" | ForEach-Object { $_.Line.Split()[-1] }
    return $buckets
}

function Show-Dashboard {
    Clear-Host
    Write-Host ""
    Write-Host "  ╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "  ║        MISSION 9 - AWS INFRASTRUCTURE MONITOR                  ║" -ForegroundColor Cyan
    Write-Host "  ╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Last update: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
    Write-Host "  Refresh: every ${refreshInterval}s | Press Ctrl+C to exit" -ForegroundColor Gray
    Write-Host ""
    
    # EMR Status
    Write-Host "  ┌─────────────────────────────────────────────────────────────────┐" -ForegroundColor White
    Write-Host "  │  EMR CLUSTER                                                    │" -ForegroundColor White
    Write-Host "  ├─────────────────────────────────────────────────────────────────┤" -ForegroundColor White
    
    $emr = Get-EMRStatus
    if ($emr) {
        $stateColor = switch ($emr.State) {
            "WAITING" { "Green" }
            "RUNNING" { "Green" }
            "STARTING" { "Yellow" }
            "BOOTSTRAPPING" { "Yellow" }
            "TERMINATING" { "Red" }
            "TERMINATED_WITH_ERRORS" { "Red" }
            default { "White" }
        }
        Write-Host "  │  ID:     " -NoNewline; Write-Host "$($emr.Id)" -ForegroundColor Cyan
        Write-Host "  │  Name:   " -NoNewline; Write-Host "$($emr.Name)" -ForegroundColor White
        Write-Host "  │  State:  " -NoNewline; Write-Host "$($emr.State)" -ForegroundColor $stateColor
        if ($emr.Reason) {
            Write-Host "  │  Reason: " -NoNewline; Write-Host "$($emr.Reason)" -ForegroundColor Gray
        }
        if ($emr.MasterDns) {
            Write-Host "  │  Master: " -NoNewline; Write-Host "$($emr.MasterDns)" -ForegroundColor Green
            Write-Host "  │" -ForegroundColor White
            Write-Host "  │  JupyterHub: " -NoNewline; Write-Host "https://$($emr.MasterDns):9443" -ForegroundColor Cyan
        }
    } else {
        Write-Host "  │  No active cluster found" -ForegroundColor Gray
    }
    Write-Host "  └─────────────────────────────────────────────────────────────────┘" -ForegroundColor White
    Write-Host ""
    
    # S3 Status
    Write-Host "  ┌─────────────────────────────────────────────────────────────────┐" -ForegroundColor White
    Write-Host "  │  S3 BUCKET                                                      │" -ForegroundColor White
    Write-Host "  ├─────────────────────────────────────────────────────────────────┤" -ForegroundColor White
    
    $buckets = Get-S3Buckets
    if ($buckets) {
        foreach ($bucket in $buckets) {
            Write-Host "  │  " -NoNewline; Write-Host "✓ $bucket" -ForegroundColor Green
            # Count objects
            $count = docker compose run --rm aws s3 ls "s3://$bucket/Test/" --recursive 2>$null | Measure-Object | Select-Object -ExpandProperty Count
            if ($count -gt 0) {
                Write-Host "  │    Images in /Test/: " -NoNewline; Write-Host "$count" -ForegroundColor Cyan
            }
        }
    } else {
        Write-Host "  │  No mission9 bucket found" -ForegroundColor Gray
    }
    Write-Host "  └─────────────────────────────────────────────────────────────────┘" -ForegroundColor White
    Write-Host ""
    
    # Progress indicator
    if ($emr) {
        Write-Host "  Progress: " -NoNewline
        $stages = @("STARTING", "BOOTSTRAPPING", "RUNNING", "WAITING")
        $currentIndex = $stages.IndexOf($emr.State)
        for ($i = 0; $i -lt $stages.Count; $i++) {
            if ($i -lt $currentIndex) {
                Write-Host "[$($stages[$i])] " -NoNewline -ForegroundColor Green
            } elseif ($i -eq $currentIndex) {
                Write-Host "[$($stages[$i])] " -NoNewline -ForegroundColor Yellow
            } else {
                Write-Host "[$($stages[$i])] " -NoNewline -ForegroundColor Gray
            }
        }
        Write-Host ""
    }
}

# Main loop
Set-Location "C:\git\mission9"
while ($true) {
    Show-Dashboard
    Start-Sleep -Seconds $refreshInterval
}
