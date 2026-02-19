# Deployment Guide - Multimodal Emotion Recognition System

## Table of Contents
1. [Local Deployment](#local-deployment)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [Environment Variables](#environment-variables)
5. [Troubleshooting](#troubleshooting)

## Local Deployment

### Prerequisites
- Python 3.8+
- pip
- Virtual environment (recommended)

### Quick Start
```bash
# 1. Run automated setup
python setup_environment.py

# 2. Activate virtual environment
source activate.sh  # Linux/Mac
# OR
activate.bat        # Windows

# 3. Download datasets
python download_datasets.py

# 4. Train models (after downloading datasets)
python src/facial_recognition/train.py
python src/speech_analysis/train.py
python src/text_analysis/train.py

# 5. Run dashboard
python run_dashboard.py
```

### Manual Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Run dashboard
python run_dashboard.py
```

## Docker Deployment

### Build and Run with Docker

**Option 1: Docker Compose (Recommended)**
```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

**Option 2: Docker CLI**
```bash
# Build image
docker build -t emotion-recognition .

# Run container
docker run -d \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/saved_models:/app/saved_models \
  --name emotion-recognition \
  emotion-recognition

# View logs
docker logs -f emotion-recognition

# Stop container
docker stop emotion-recognition
docker rm emotion-recognition
```

### Docker with Pre-trained Models
```bash
# Copy trained models to saved_models/ directory first
# Then build and run
docker-compose up -d
```

## Cloud Deployment

### Streamlit Cloud (Easiest)

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select `run_dashboard.py` as the main file
   - Click "Deploy"

3. **Add Secrets (if needed):**
   - In Streamlit Cloud dashboard, go to Settings → Secrets
   - Add any API keys or credentials

### AWS EC2 Deployment

1. **Launch EC2 Instance:**
   - Ubuntu 20.04 LTS
   - t2.medium or larger (for model inference)
   - Open port 8501 in security group

2. **Connect and Setup:**
   ```bash
   # SSH into instance
   ssh -i your-key.pem ubuntu@your-ec2-ip

   # Install Docker
   sudo apt update
   sudo apt install -y docker.io docker-compose
   sudo usermod -aG docker ubuntu

   # Clone repository
   git clone <your-repo-url>
   cd "Student's Emotion Recognition using Multimodality and Deep Learning"

   # Deploy with Docker
   docker-compose up -d
   ```

3. **Access Dashboard:**
   - Navigate to `http://your-ec2-ip:8501`

### Google Cloud Run

1. **Build and Push Image:**
   ```bash
   # Set project
   gcloud config set project YOUR_PROJECT_ID

   # Build image
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/emotion-recognition

   # Deploy
   gcloud run deploy emotion-recognition \
     --image gcr.io/YOUR_PROJECT_ID/emotion-recognition \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --port 8501
   ```

2. **Access:**
   - Cloud Run will provide a URL

### Azure App Service

1. **Create App Service:**
   ```bash
   # Login
   az login

   # Create resource group
   az group create --name emotion-recognition-rg --location eastus

   # Create App Service plan
   az appservice plan create \
     --name emotion-recognition-plan \
     --resource-group emotion-recognition-rg \
     --is-linux \
     --sku B2

   # Create web app
   az webapp create \
     --resource-group emotion-recognition-rg \
     --plan emotion-recognition-plan \
     --name emotion-recognition-app \
     --deployment-container-image-name emotion-recognition:latest
   ```

## Environment Variables

Create a `.env` file for configuration:

```bash
# Application Settings
APP_NAME=Multimodal Emotion Recognition
APP_PORT=8501

# Model Paths
FACIAL_MODEL_PATH=saved_models/facial_emotion_model.h5
SPEECH_MODEL_PATH=saved_models/speech_emotion_model.h5
TEXT_MODEL_PATH=saved_models/text_emotion_model.pt

# Fusion Configuration
FUSION_TYPE=weighted
FACIAL_WEIGHT=0.4
SPEECH_WEIGHT=0.3
TEXT_WEIGHT=0.3

# API Keys (if using external services)
# GOOGLE_API_KEY=your_key_here
# WHISPER_API_KEY=your_key_here
```

## Performance Optimization

### For Production Deployment

1. **Use GPU (if available):**
   ```dockerfile
   # In Dockerfile, use CUDA base image
   FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
   ```

2. **Model Optimization:**
   - Convert models to TensorFlow Lite or ONNX
   - Use quantization for smaller model size
   - Enable model caching

3. **Resource Limits:**
   ```yaml
   # In docker-compose.yml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 4G
       reservations:
         cpus: '1'
         memory: 2G
   ```

## Monitoring and Logging

### Application Logs
```bash
# Docker logs
docker-compose logs -f

# Save logs to file
docker-compose logs > app.log
```

### Health Checks
```bash
# Check if app is running
curl http://localhost:8501/_stcore/health
```

### Metrics (Optional)
Add Prometheus/Grafana for monitoring:
- Request counts
- Response times
- Model inference latency
- Error rates

## Security Best Practices

1. **Use HTTPS in production:**
   - Set up SSL/TLS certificates
   - Use reverse proxy (nginx)

2. **Implement authentication:**
   - Add user login system
   - Use OAuth or JWT tokens

3. **Rate limiting:**
   - Prevent API abuse
   - Use nginx or application-level rate limiting

4. **Secure secrets:**
   - Never commit API keys
   - Use environment variables or secret managers

## Scaling

### Horizontal Scaling
```yaml
# docker-compose.yml
services:
  emotion-recognition:
    deploy:
      replicas: 3
```

### Load Balancing
Use nginx as reverse proxy:
```nginx
upstream emotion_app {
    server localhost:8501;
    server localhost:8502;
    server localhost:8503;
}

server {
    listen 80;
    location / {
        proxy_pass http://emotion_app;
    }
}
```

## Troubleshooting

### Common Issues

**Issue: Port already in use**
```bash
# Find process using port 8501
lsof -i :8501
# Kill process
kill -9 <PID>
```

**Issue: Out of memory**
- Reduce batch size in config.py
- Use smaller model architectures
- Increase Docker memory limit

**Issue: Models not loading**
- Verify model files exist in saved_models/
- Check file permissions
- Ensure correct paths in config.py

**Issue: Slow inference**
- Use GPU if available
- Optimize models (quantization)
- Enable model caching
- Use smaller input sizes

## Backup and Recovery

### Backup Models
```bash
# Backup trained models
tar -czf models_backup.tar.gz saved_models/

# Restore
tar -xzf models_backup.tar.gz
```

### Database Backup (if using)
```bash
# Backup prediction history
docker exec emotion-recognition pg_dump -U user db > backup.sql
```

## CI/CD Pipeline

Example GitHub Actions workflow:
```yaml
name: Deploy
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build and push Docker image
        run: |
          docker build -t emotion-recognition .
          docker push your-registry/emotion-recognition
      - name: Deploy to production
        run: |
          # Your deployment commands
```

---

**Need Help?** Check the [USER_GUIDE.md](USER_GUIDE.md) or [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)
