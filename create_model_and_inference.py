import sys
import os
import argparse
from sagemaker.transformer import Transformer
from sagemaker.pytorch import PyTorchModel


    
def parse_args(args):
    """
    create model or serve by model name
    send params as command line args
    model_name - name of the model to create or serve
    image_uri - image to use for the model
    model_artifact_path - path to the model artifact
    batch_size : int
    tokenizer_max_length : int
    sagemaker stradegy§: 'MultiRecord' or 'SingleRecord'
    accept: 'text/csv'
    instance_type: str
    instance_count: int
    Data location:
    input_data: str (s3://bucket/prefix)
    output_data: str (s3://bucket/prefix)
    join_source: 'Input' or 'None'
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--sg_model_name', type=str, default=None)
    parser.add_argument('--model_artifact_path', type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--strategy', type=str, default='MultiRecord')
    parser.add_argument('--accept', type=str, default='text/csv')
    parser.add_argument('--assemble_with', type=str, default='Line')
    parser.add_argument('--instance_type', type=str, default='ml.g4dn.xlarge')
    parser.add_argument('--instance_count', type=int, default=1)
    parser.add_argument('--input_data', type=str)
    parser.add_argument('--output_data', type=str, default=None)
    parser.add_argument('--join_source', type=str, default=None)
    parser.add_argument("--create_model", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--st_model", type=str, default=None)
    return parser.parse_args()



def create_sagemaker_model(args):
    try:

        pytorch_model = PyTorchModel(
            name=args.sg_model_name,
            model_data=args.model_artifact_path,
            role=os.environ.get("SAGEMAKER_ROLE"),
            framework_version="2.0.0",
            py_version="py310",
            source_dir="code",
            entry_point="inference.py",
            env={"MODEL_NAME": args.st_model or 'sentence-transformers/LaBSE'},
        )
        pytorch_model = pytorch_model.transformer(
        instance_count=args.instance_count, 
        instance_type=args.instance_type,
        accept=args.accept,
        )

    
    except Exception as e:
        print(f'error creating model {e}')
        
    

def serve(args):
    transformer = Transformer(
        model_name=args.sg_model_name,
        output_path=args.output_data or args.input_data,
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        accept=args.accept,
        assemble_with=args.assemble_with,
        strategy=args.strategy,
        max_payload=70,
        env={
            'BATCH_SIZE': str(args.batch_size),
            'STRATEGY': args.strategy,
            'MODEL_NAME': args.st_model or 'sentence-transformers/LaBSE'}
    )
    print(f'Starting transform job')
    transformer.transform(
        args.input_data,
        content_type=args.accept,
        join_source=args.join_source,
        split_type=args.assemble_with,
        wait=False
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    args = parse_args(args)
    if args.create_model:
        if args.model_artifact_path is None:
            print("model_artifact_path is required")
            sys.exit(1)
        create_sagemaker_model(args)
    if args.serve or not args.create_model:
        if args.input_data is None:
            print("input_data is required")
            sys.exit(1)
        serve(args)


