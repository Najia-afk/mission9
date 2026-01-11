#!/bin/sh
# Auto-update status.json every 10 seconds
CLUSTER_ID="j-27K4EIH03ZF1X"
REGION="eu-west-1"

echo "=== Status Updater Started ==="
echo "Cluster: $CLUSTER_ID"
echo "Updating every 10 seconds..."
echo ""

while true; do
    aws emr describe-cluster --cluster-id $CLUSTER_ID --region $REGION > /app/infra/status.json 2>&1
    STATE=$(grep -o '"State": "[^"]*"' /app/infra/status.json | head -1 | cut -d'"' -f4)
    echo "[$(date +%H:%M:%S)] $STATE"
    
    if [ "$STATE" = "WAITING" ]; then
        DNS=$(grep -o '"MasterPublicDnsName": "[^"]*"' /app/infra/status.json | cut -d'"' -f4)
        echo ""
        echo "✅ CLUSTER READY!"
        echo "JupyterHub: https://$DNS:9443"
        echo "Login: jovyan / jupyter"
    fi
    
    sleep 10
done
