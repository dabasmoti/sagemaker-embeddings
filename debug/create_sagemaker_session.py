import boto3
import sagemaker
from sagemaker.local import LocalSession

def get_sagemaker_session(mode=None):
    if mode == 'local':
        print('running in local mode')
        sagemaker_session = LocalSession()
        sagemaker_session.config = {'local': {'local_code': True}}
        return sagemaker_session
    else:
        session = boto3.Session(
        # aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
        # aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
        # region_name=Config.AWS_DEFAULT_REGION
        )
        return sagemaker.Session(session)
