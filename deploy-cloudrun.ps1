# Configuration
$ProjectId = "the-nyx-ai"
$Region = "europe-west2"
$ServiceName = "avatar-chat-server"
$ImageName = "gcr.io/$ProjectId/$ServiceName"

# Authenticate
Write-Host "Authenticating with GCP..."
gcloud auth login
gcloud config set project $ProjectId
gcloud auth configure-docker gcr.io --quiet

# Build locally
Write-Host "Building Docker image locally..."
docker build -t $ImageName .

# Push to GCR
Write-Host "Pushing image to GCR..."
docker push $ImageName

# Deploy to Cloud Run
Write-Host "Deploying to Cloud Run..."

# Convert .env to env.yaml for Cloud Run
if (Test-Path ".env") {
    Get-Content ".env" | Where-Object { $_ -match "^\w+=" } | ForEach-Object {
        $parts = $_ -split "=", 2
        $value = $parts[1] -replace '^"|"$', ''  # Remove surrounding quotes
        $value = $value -replace '"', '\"'        # Escape inner quotes
        "$($parts[0]): '$value'"
    } | Set-Content "env.yaml"
}

gcloud run deploy $ServiceName `
  --image $ImageName `
  --region $Region `
  --platform managed `
  --allow-unauthenticated `
  --port 8080 `
  --cpu=1 `
  --memory=1536Mi `
  --env-vars-file env.yaml

Write-Host "Deployment complete!"
