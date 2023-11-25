import os
from config import Config
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.transformer import Transformer
import boto3
import sagemaker
from sagemaker.local import LocalSession
from sagemaker.pytorch import PyTorchModel


def get_sagemaker_session(mode=None):
    if mode == "local":
        print("running in local mode")
        sagemaker_session = LocalSession()
        sagemaker_session.config = {"local": {"local_code": True}}
        return sagemaker_session
    else:
        session = boto3.Session()
        return sagemaker.Session(session)


class BatchTransformManager:
    def __init__(self, role):
        self.model_name = None
        self.model_path = None
        self.role = role
        self.session = None

    def create_model(
        self,
        model_name: str,
        model_path: str,
        environment_vars: dict = None,
        pytorch_version: str = "2.0.0",
        py_version: str = "py310",
        source_dir: str = "code",
        entry_point: str = "inference.py",
        sagemaker_session=None,
    ):
        self.model_name = model_name
        self.model_path = model_path
        print("creating sagemaker model")

        sagemaker_model = PyTorchModel(
            name=self.model_name,
            model_data=self.model_path,
            role=self.role,
            source_dir=source_dir,
            framework_version=pytorch_version,
            py_version=py_version,
            entry_point=entry_point,
            env=environment_vars,
            sagemaker_session=sagemaker_session,
        )

        return sagemaker_model

    def run_batch_job(
        self,
        model_name: str = None,
        model_path: str = None,
        input_path: str = None,
        output_path: str = None,
        instance_count: int = 1,
        instance_type: str = "ml.g4dn.xlarge",
        strategy: str = "MultiRecord",
        accept: str = "text/csv",
        split_type: str = "Line",
        envs: dict = None,
    ):
        
        self.session = get_sagemaker_session(instance_type)
        sage_maker_model = self.create_model(
            model_name=model_name, 
            model_path=model_path, 
            sagemaker_session=self.session,
            environment_vars=envs,
        )

        transformer = sage_maker_model.transformer(
            output_path=output_path or input_path,
            instance_count=instance_count,
            instance_type=instance_type,
            accept=accept,
            assemble_with=split_type,
            strategy=strategy,
            max_payload=70,
            env=envs,
        )

        transformer.transform(
            input_path, content_type=accept, split_type=split_type, wait=False
        )


if __name__ == "__main__":
    import os
    sgm = BatchTransformManager(
        role=os.environ.get("SAGEMAKER_ROLE"),
    )
    print("role",os.environ.get("SAGEMAKER_ROLE"))
    sgm.run_batch_job(
        model_path="s3://llama-weights-us/dummy-model/model.tar.gz",
        input_path="s3://llama-weights-us/input_data/",
        envs={"MODEL_NAME": "all-MiniLM-L12-v2"},
        instance_type='local'
    )
    
