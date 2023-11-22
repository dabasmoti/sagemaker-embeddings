import os

class Config:
    AWS_DEFAULT_REGION = "us-east-1"

    SG_ROLE = 'arn:aws:iam::834250429826:role/mpa-sagemaker'
    SG_TRANSFORMERS_VERSION = '4.6'
    SG_PYTORCH_VERSION = '1.7'
    SG_PY_VERSION = 'py36'
    SG_ENTRY_POINT = 'inference.py'
    SG_CONTENT_TYPE = 'text/csv'
    SG_SPLIT_TYPE = 'Line'
    SG_STRATEGY = 'MultiRecord'
    SG_TEXT_FIELD_KEY = 'inputs'
    SG_INSTANCE_TYPE = 'ml.m4.xlarge'
    SG_INSTANCE_COUNT = 1
    SG_PREDICTION_COL = 'label'
    SG_JOIN_SOURCE = 'Input'
    SG_SOURCE_DIR = 'code'






