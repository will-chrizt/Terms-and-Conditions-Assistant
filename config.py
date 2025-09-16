import os

# AWS + Model config
AWS_REGION = "us-east-1"
EMBED_MODEL = "amazon.titan-embed-text-v2:0"
LLM_MODEL = "anthropic.claude-3-5-sonnet-20240620-v1:0"

# Ensure AWS region is set
os.environ["AWS_REGION_NAME"] = AWS_REGION
