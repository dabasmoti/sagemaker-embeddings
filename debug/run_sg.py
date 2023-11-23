from datetime import datetime
from batch_manager import BatchTransformManager



def generate_str_datetime():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

bm = BatchTransformManager(s3_uri='s3://llama-weights-us/input_data/',
                            output_s3_path='s3://llama-weights-us/input_data/',
                           model_path='s3://llama-weights-us/labse-model/model.tar.gz',
                           batch_job_name=f'batch-job-{generate_str_datetime()}',
                           instance_count=1,
                           instance_type='local',
                           env)
bm.create_and_run_batch_job()