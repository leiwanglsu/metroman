#!/bin/bash
#
# Script to deploy Terraform and Docker image AWS infrastructure
#
# REQUIRES:
#   jq (https://jqlang.github.io/jq/)
#   docker (https://docs.docker.com/desktop/) > version Docker 1.5
#   AWS CLI (https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
#   Terraform (https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli)
#
# Command line arguments:
# [1] registry: Registry URI
# [2] repository: Name of repository to create
# [3] prefix: Prefix to use for AWS resources associated with environment deploying to
# [4] s3_state_bucket: Name of the S3 bucket to store Terraform state in (no need for s3:// prefix)
# [5] profile: Name of profile used to authenticate AWS CLI commands
# 
# Example usage: ./deploy.sh "account-id.dkr.ecr.region.amazonaws.com" "container-image-name" "prefix-for-environment" "s3-state-bucket-name" "confluence-named-profile" 

REGISTRY=$1
REPOSITORY=$2
PREFIX=$3
S3_STATE=$4
PROFILE=$5


# Deploy Container Image
./deploy-ecr.sh $REGISTRY $REPOSITORY $PREFIX $PROFILE

# Deploy Terraform
cd terraform/
terraform init -reconfigure -backend-config="bucket=$S3_STATE" -backend-config="key=metroman.tfstate" -backend-config="region=us-west-2" -backend-config="profile=$PROFILE"
terraform apply -var-file="conf.tfvars" -auto-approve
cd ..