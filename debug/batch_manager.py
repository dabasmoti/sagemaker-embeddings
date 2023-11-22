import os
from config import Config
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.transformer import Transformer
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
        )
        return sagemaker.Session(session)


class BatchTransformManager:

    def __init__(self, role, session=None):
        self.model_name = None
        self.model_path = None
        self.role = role    
        self.session = get_sagemaker_session(session)
 

    def create_model(
            self,
            model_name: str,
            model_path: str,
            environment_vars: dict = None,
            transformers_version: str = '4.6',
            pytorch_version: str = '1.7',
            py_version: str = 'py36',
            source_dir: str =  'code',
            entry_point: str = 'inference.py',
            sagemaker_session = None,
    ):
        self.model_name = model_name
        self.model_path = model_path
        print('creating sagemaker model')

        sagemaker_model = HuggingFaceModel(
            # name=self.model_name,
            model_data=self.model_path,
            role=self.role,
            source_dir=source_dir,
            transformers_version=transformers_version,
            pytorch_version=pytorch_version,
            py_version=py_version,
            entry_point=entry_point,
            env=environment_vars,
            sagemaker_session=sagemaker_session
        )
        
        return sagemaker_model

    def run_batch_job(
            self,
            model_name: str=None,
            model_path: str=None,
            input_path: str=None,
            output_path: str=None,
            instance_count: int=1,
            instance_type: str='local',
            strategy: str='MultiRecord',
            accept: str='text/csv',
            split_type: str='Line',
            envs: dict=None):
        
        sage_maker_model = self.create_model(model_name=model_name, model_path=model_path, sagemaker_session=self.session)
    
        transformer = sage_maker_model.transformer(
            output_path=output_path or input_path,
            instance_count=instance_count,
            instance_type=instance_type,
            accept=accept,
            assemble_with=split_type,
            strategy=strategy,
            max_payload=70,
            env=envs
        )

        transformer.transform(
        input_path,
        content_type=accept,
        split_type=split_type,
        wait=False
    )
        

if __name__ == "__main__":
    sgm = BatchTransformManager(role="arn:aws:iam::834250429826:role/service-role/AmazonSageMaker-ExecutionRole-20231122T172883", session='local')    
    sgm.run_batch_job(model_path='s3://llama-weights-us/labse-model/model.tar.gz' ,input_path='s3://llama-weights-us/input_data/')
    # run_batch_job()