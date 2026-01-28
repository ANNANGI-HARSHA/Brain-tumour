# Brain Tumor Analyzer - Google Cloud Deployment Guide

## 🚀 Deployment Options

### Option 1: Google Cloud Run (Recommended)

1. **Install Google Cloud SDK**:
   ```bash
   # Download from: https://cloud.google.com/sdk/docs/install
   ```

2. **Authenticate**:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

3. **Build and Deploy**:
   ```bash
   # Build container
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/brain-tumor-analyzer
   
   # Deploy to Cloud Run
   gcloud run deploy brain-tumor-analyzer \
     --image gcr.io/YOUR_PROJECT_ID/brain-tumor-analyzer \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 4Gi \
     --cpu 2 \
     --timeout 300
   ```

### Option 2: Google App Engine

1. **Deploy**:
   ```bash
   gcloud app deploy app.yaml
   ```

2. **View App**:
   ```bash
   gcloud app browse
   ```

### Option 3: Google Compute Engine

1. **Create VM Instance**:
   ```bash
   gcloud compute instances create brain-tumor-analyzer \
     --image-family=ubuntu-2004-lts \
     --image-project=ubuntu-os-cloud \
     --machine-type=n1-standard-2 \
     --zone=us-central1-a
   ```

2. **SSH and Setup**:
   ```bash
   gcloud compute ssh brain-tumor-analyzer
   
   # Install dependencies
   sudo apt-get update
   sudo apt-get install python3-pip
   pip3 install -r requirements.txt
   
   # Run application
   streamlit run app/streamlit_app.py --server.port 8080
   ```

## 📋 Pre-Deployment Checklist

- [ ] Model file exists at `saved_models/classifier_best.pth`
- [ ] All dependencies listed in `requirements.txt`
- [ ] Environment variables configured
- [ ] Docker tested locally: `docker build -t brain-tumor-analyzer .`
- [ ] Resource limits appropriate for expected traffic
- [ ] Security settings reviewed

## 🔐 Security Recommendations

1. Enable HTTPS (automatic with Cloud Run)
2. Add authentication for production use
3. Set up Cloud Armor for DDoS protection
4. Enable Cloud Logging and Monitoring
5. Configure firewall rules

## 📊 Monitoring

```bash
# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=brain-tumor-analyzer"

# Monitor performance
gcloud monitoring dashboards list
```

## 🎯 Performance Optimization

1. **Enable CDN**: For static assets
2. **Use Cloud Storage**: For large model files
3. **Auto-scaling**: Configure based on traffic patterns
4. **Caching**: Implement Redis for repeated analyses

## 💰 Cost Estimation

- **Cloud Run**: ~$0.00002 per request (1M requests free/month)
- **App Engine**: ~$0.05 per instance-hour
- **Compute Engine**: ~$50/month (n1-standard-2)

## 🆘 Troubleshooting

**Memory Issues**: Increase `--memory` flag
**Timeout Errors**: Increase `--timeout` value
**Permission Denied**: Check IAM roles and service accounts

## 📞 Support

For deployment issues, check:
- Google Cloud Console logs
- Application logs in Streamlit
- Network configuration

✅ Training Complete!
======================================================================
  Time: 180.45 minutes
  Best Validation Accuracy: 92.34%

🎉 SUCCESS! Achieved 92.34% >= 90% target!
