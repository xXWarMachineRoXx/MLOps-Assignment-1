#!/usr/bin/env bash
set -euo pipefail

# 0) Set context
SUBSCRIPTION="b5122c2a-6560-4ec8-8af8-533db897917b"
RG="rg-mlopsassignment1"
LOCATION="centralindia"
ACR="mlopsassignmentregisty"
CLUSTER="MLOPsAssignment1Cluster"
NAMESPACE="heart-disease"
IMAGE_TAG="heart-disease-api:1.0"
REPO="https://github.com/xXWarMachineRoXx/MLOps-Assignment-1.git"
BRANCH="main"
CONTEXT_PATH="."
# ordered by preference; script will pick the first available in region/subscription
PREFERRED_SIZES=(
  "Standard_D2_v5"
  "Standard_D2_v6"
)
# set SKIP_BUILD=1 to reuse existing image tag
SKIP_BUILD=${SKIP_BUILD:-0}
# optional DNS settings (leave empty to skip)
# Custom domain via Azure DNS zone (requires delegation at registrar)
DNS_ZONE=""          # e.g., example.com
DNS_RECORD=""        # e.g., api (results in api.example.com)
DNS_TTL=300
DNS_RG="$RG"         # DNS resource group (defaults to main RG)

# Default Azure-generated hostname using a static Public IP with DNS label under *.cloudapp.azure.com (incurs public IP charges)
DEFAULT_DNS_LABEL="" # e.g., heart-api-ci -> heart-api-ci.<region>.cloudapp.azure.com
DEFAULT_PIP_NAME="heart-disease-pip"

# FQDN for deployment image reference
IMAGE_FQDN="${ACR}.azurecr.io/${IMAGE_TAG}"

az account set --subscription "$SUBSCRIPTION"

# 1) Resource group
echo "[info] ensuring resource group $RG in $LOCATION"
az group create -n "$RG" -l "$LOCATION"

# 2) ACR (idempotent)
echo "[info] ensuring ACR $ACR"
if ! az acr show -n "$ACR" -g "$RG" >/dev/null 2>&1; then
  az acr create -n "$ACR" -g "$RG" --sku Standard --admin-enabled false
else
  echo "ACR $ACR already exists"
fi

# 2b) Choose the first available VM size from preference list in this region
echo "[info] selecting available VM size in $LOCATION"
CHOSEN_SIZE=""
AVAILABLE_SIZES=()
# Optimistic check in order; if creation fails we'll see the error
for SIZE in "${PREFERRED_SIZES[@]}"; do
  echo "[debug] attempting preferred SKU $SIZE"
  AVAILABLE_SIZES+=("$SIZE")
  CHOSEN_SIZE="$SIZE"
  echo "Using node size: $CHOSEN_SIZE"
  break
done

if [[ -z "$CHOSEN_SIZE" ]]; then
  echo "No preferred VM sizes available in $LOCATION." >&2
  echo "Preferred list: ${PREFERRED_SIZES[*]}" >&2
  echo "Sample of available VM sizes in $LOCATION:" >&2
  az vm list-skus --location "$LOCATION" --resource-type virtualMachines \
    --query "[].name" -o tsv | sort -u | head -n 25 >&2
  echo "Update PREFERRED_SIZES to one of the available sizes and retry." >&2
  exit 1
fi

echo "[info] available from preferred list: ${AVAILABLE_SIZES[*]}"

# 3) Build & push image from GitHub repo (server-side build, no local Docker needed)
if [[ "$SKIP_BUILD" == "1" ]]; then
  echo "[info] skipping build (SKIP_BUILD=1). Ensure $IMAGE_FQDN exists."
else
  echo "[info] building and pushing image $IMAGE_FQDN from $REPO ($BRANCH)"
  az acr build \
    --registry "$ACR" \
    --image "$IMAGE_TAG" \
    "${REPO}#${BRANCH}:${CONTEXT_PATH}"
fi

# 4) AKS (managed identity, ACR attached) idempotent create
echo "[info] ensuring AKS cluster $CLUSTER"
if ! az aks show -n "$CLUSTER" -g "$RG" >/dev/null 2>&1; then
  az aks create -n "$CLUSTER" -g "$RG" -l "$LOCATION" \
    --node-vm-size "$CHOSEN_SIZE" --node-count 2 \
    --enable-managed-identity --attach-acr "$ACR" --generate-ssh-keys
else
  echo "AKS $CLUSTER already exists"
  # ensure ACR is attached
  az aks update -n "$CLUSTER" -g "$RG" --attach-acr "$ACR" >/dev/null 2>&1 || true
fi

# capture node resource group for networking resources
NODE_RG=$(az aks show -n "$CLUSTER" -g "$RG" --query nodeResourceGroup -o tsv)
echo "[info] AKS node resource group: $NODE_RG"

# 5) Get kubeconfig
echo "[info] fetching kubeconfig for $CLUSTER"
az aks get-credentials -n "$CLUSTER" -g "$RG"

# 6) Namespace (idempotent)
echo "[info] ensuring namespace $NAMESPACE"
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

# 7) Deploy manifest in-line (uses IMAGE_FQDN)
echo "[info] applying deployment and service"
cat <<EOF | kubectl apply -n "$NAMESPACE" -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: heart-disease-api
  labels:
    app: heart-disease-api
spec:
  replicas: 2
  revisionHistoryLimit: 3
  selector:
    matchLabels:
      app: heart-disease-api
  template:
    metadata:
      labels:
        app: heart-disease-api
    spec:
      containers:
      - name: api
        image: ${IMAGE_FQDN}
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: heart-disease-service
spec:
  type: LoadBalancer
  selector:
    app: heart-disease-api
  ports:
  - port: 80
    targetPort: 8000
EOF

# 8) Check status & get external IP
echo "[info] current workloads in namespace $NAMESPACE"
kubectl get deploy,po,svc -n "$NAMESPACE"

# 9a) Optional default Azure DNS label via static Public IP (cloudapp.azure.com)
EXTERNAL_IP=""
STATIC_IP=""
if [[ -n "$DEFAULT_DNS_LABEL" ]]; then
  echo "[info] ensuring static Public IP $DEFAULT_PIP_NAME with DNS label $DEFAULT_DNS_LABEL (charges may apply)"
  if ! az network public-ip show -g "$NODE_RG" -n "$DEFAULT_PIP_NAME" >/dev/null 2>&1; then
    az network public-ip create -g "$NODE_RG" -n "$DEFAULT_PIP_NAME" \
      --sku Standard --allocation-method static --dns-name "$DEFAULT_DNS_LABEL"
  else
    echo "[info] Public IP $DEFAULT_PIP_NAME already exists"
  fi
  STATIC_IP=$(az network public-ip show -g "$NODE_RG" -n "$DEFAULT_PIP_NAME" --query ipAddress -o tsv)
  echo "[info] static IP: $STATIC_IP"
  kubectl annotate svc heart-disease-service -n "$NAMESPACE" \
    service.beta.kubernetes.io/azure-load-balancer-resource-group="$NODE_RG" --overwrite
  kubectl patch svc heart-disease-service -n "$NAMESPACE" -p \
    "{\"spec\": {\"loadBalancerIP\": \"$STATIC_IP\"}}"
  echo "[info] waiting for LoadBalancer to honor static IP"
  for i in {1..30}; do
    EXTERNAL_IP=$(kubectl get svc heart-disease-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)
    [[ -n "$EXTERNAL_IP" ]] && break
    sleep 5
  done
  if [[ -z "$EXTERNAL_IP" ]]; then
    echo "[warn] load balancer IP not assigned yet; DNS label may not be active" >&2
  else
    DEFAULT_FQDN="${DEFAULT_DNS_LABEL}.${LOCATION}.cloudapp.azure.com"
    echo "[info] external IP: $EXTERNAL_IP"
    echo "[info] default DNS: $DEFAULT_FQDN"
  fi
else
  echo "[info] DEFAULT_DNS_LABEL not set; skipping default cloudapp DNS"
fi

# 9b) Optional DNS A record for a custom zone
if [[ -n "$DNS_ZONE" && -n "$DNS_RECORD" ]]; then
  echo "[info] waiting for LoadBalancer external IP"
  for i in {1..30}; do
    [[ -n "$EXTERNAL_IP" ]] && break
    EXTERNAL_IP=$(kubectl get svc heart-disease-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)
    [[ -n "$EXTERNAL_IP" ]] && break
    sleep 5
  done
  if [[ -z "$EXTERNAL_IP" ]]; then
    echo "[warn] external IP not assigned yet; skipping DNS update" >&2
  else
    echo "[info] external IP: $EXTERNAL_IP"
    echo "[info] ensuring DNS zone $DNS_ZONE in $DNS_RG"
    if ! az network dns zone show -g "$DNS_RG" -n "$DNS_ZONE" >/dev/null 2>&1; then
      az network dns zone create -g "$DNS_RG" -n "$DNS_ZONE"
    fi
    echo "[info] upserting A record $DNS_RECORD.$DNS_ZONE -> $EXTERNAL_IP"
    az network dns record-set a create -g "$DNS_RG" -z "$DNS_ZONE" -n "$DNS_RECORD" --ttl "$DNS_TTL" >/dev/null 2>&1 || true
    az network dns record-set a add-record -g "$DNS_RG" -z "$DNS_ZONE" -n "$DNS_RECORD" -a "$EXTERNAL_IP" --ttl "$DNS_TTL" --if-none-match >/dev/null 2>&1 || \
    az network dns record-set a update -g "$DNS_RG" -z "$DNS_ZONE" -n "$DNS_RECORD" --set "arecords=[{ipv4Address:$EXTERNAL_IP}]" --ttl "$DNS_TTL"
    echo "[info] DNS record set: $DNS_RECORD.$DNS_ZONE -> $EXTERNAL_IP (TTL $DNS_TTL)"
  fi
else
  echo "[info] DNS_ZONE or DNS_RECORD not set; skipping custom DNS update"
fi