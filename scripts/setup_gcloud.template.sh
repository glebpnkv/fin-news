#!/bin/bash
# setup_gcloud.sh - Run this to use this project with your GCP account

# Create a named configuration for this project
gcloud config configurations create fin-news

# Activate it
gcloud config configurations activate fin-news

PROJECT_ID="your-gcp-project-id"
REGION="us-central1"

gcloud auth login
echo "Setting up gcloud for fin-news project..."
gcloud config set project $PROJECT_ID
gcloud config set compute/region $REGION

echo "Current configuration:"
gcloud config list