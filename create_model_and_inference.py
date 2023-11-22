import sys
import argparse
from sagemaker.transformer import Transformer
import boto3
from debug.create_sagemaker_session

role = "arn:aws:iam::834250429826:role/service-role/AmazonSageMaker-ExecutionRole-20231122T172883"
    
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
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--image_uri', type=str)
    parser.add_argument('--model_artifact_path', type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--strategy', type=str, default='MultiRecord')
    parser.add_argument('--accept', type=str, default='text/csv')
    parser.add_argument('--assemble_with', type=str, default='Line')
    parser.add_argument('--instance_type', type=str, default='ml.m4.xlarge')
    parser.add_argument('--instance_count', type=int, default=1)
    parser.add_argument('--input_data', type=str)
    parser.add_argument('--output_data', type=str, default=None)
    parser.add_argument('--join_source', type=str, default='Input')
    parser.add_argument("--create_model", action="store_true")
    parser.add_argument("--serve", action="store_true")
    return parser.parse_args()



def create_sagemaker_model(args):
    sm = boto3.client("sagemaker")
    model_name = f'{args.model_name}-{args.batch_size}-{args.strategy}-{args.accept.split("/")[1]}'
    print(f'creating model {model_name}')
    try:
        sm.create_model(
            ModelName=model_name,
            ExecutionRoleArn=role,
            PrimaryContainer={
                'Image': args.image_uri,
                'ModelDataUrl': args.model_artifact_path,
                'Environment': {
                    'BATCH_SIZE': str(args.batch_size),
                    'STRATEGY': args.strategy,
                    }
            }
        )
        args.model_name =  model_name
        print(f'Model name: {args.model_name} created')
    except Exception as e:
        print(f'error creating model {e}')
        
    

def serve(args):
    transformer = Transformer(
        model_name=args.model_name,
        output_path=args.output_data or args.input_data,
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        accept=args.accept,
        assemble_with=args.assemble_with,
        strategy=args.strategy,
        max_payload=70,
        env={'MODEL_SERVER_TIMEOUT': '600'}
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


