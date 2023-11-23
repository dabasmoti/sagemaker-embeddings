import os
import io
from functools import partial
import torch
from sentence_transformers import SentenceTransformer
from datasets import Dataset
import pandas as pd

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
STRATEGY = os.environ.get('STRATEGY') or "MultiRecord"
BATCH_SIZE = int(os.environ.get('BATCH_SIZE') or 64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEXT_COLUMN = os.environ.get('TEXT_COLUMN') or 'inputs'
PREDICTION_COLUMN = os.environ.get('PREDICTION_COLUMN') or 'embeddings'
MODEL_NAME = os.environ.get('MODEL_NAME') or 'sentence-transformers/LaBSE'



def model_fn(model_dir):
    return SentenceTransformer(MODEL_NAME).to(DEVICE)


def predict(model, batch):
    embeddings = model.encode(batch[TEXT_COLUMN], convert_to_numpy=True)
    return {PREDICTION_COLUMN: embeddings}


def transform_fn(model, input_data, content_type, accept):
    predict_func = partial(predict, model)

    if STRATEGY == 'MultiRecord':
        df = pd.read_csv(io.StringIO(input_data), sep=',', header=None)
        df.columns = [TEXT_COLUMN]
        ds = Dataset.from_pandas(df)
        ds = ds.map(predict_func, batched=True, batch_size=BATCH_SIZE)
        return pd.DataFrame([[row[TEXT_COLUMN]] + row[PREDICTION_COLUMN] for row in ds]).to_csv(index=False, header=False)

    else:
        raise NotImplementedError(f"Strategy {STRATEGY} is not implemented")


if __name__ == '__main__':
    import io
    model = model_fn(None)
    data_list = ["ישראל חי עם", "עם ישראל חי", "תנו לצהל לנצח"]
    data_str = '\n'.join(data_list)
    data_str = 'inputs\n' + data_str
    data_io = io.StringIO(data_str)
    res = transform_fn(model, data_str, None, None)
    print(res)
    res.to_csv('embeddings.csv', index=False, header=False)


