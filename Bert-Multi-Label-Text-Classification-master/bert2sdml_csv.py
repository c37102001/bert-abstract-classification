import pickle
import torch
import pandas as pd

def SubmitGenerator(prediction, sampleFile, public=True, filename='prediction.csv'):
    """
    Args:
        prediction (numpy array)
        sampleFile (str)
        public (boolean)
        filename (str)
    """
    sample = pd.read_csv(sampleFile)
    submit = {}
    submit['order_id'] = list(sample.order_id.values)
    redundant = len(sample) - prediction.shape[0]
    if public:
        submit['THEORETICAL'] = list(prediction[:, 0]) + [0]*redundant
        submit['ENGINEERING'] = list(prediction[:, 1]) + [0]*redundant
        submit['EMPIRICAL'] = list(prediction[:, 2]) + [0]*redundant
        submit['OTHERS'] = list(prediction[:, 3]) + [0]*redundant
    else:
        submit['THEORETICAL'] = [0]*redundant + list(prediction[:, 0])
        submit['ENGINEERING'] = [0]*redundant + list(prediction[:, 1])
        submit['EMPIRICAL'] = [0]*redundant + list(prediction[:, 2])
        submit['OTHERS'] = [0]*redundant + list(prediction[:, 3])
    df = pd.DataFrame.from_dict(submit)
    df.to_csv(filename, index=False)


with open('./pybert/sdml/test_prob_raw.pkl', 'rb') as f:
    result = pickle.load(f)

prediction = []
for raw in result:
    raw = torch.Tensor([raw])
    raw = raw > 0.5
    prediction.append(raw)
prediction = torch.cat(prediction).detach().numpy().astype(int)

SubmitGenerator(prediction,
                './pybert/sdml/raw_data/task2_sample_submission.csv',
                True,
                './pybert/sdml/result/bert_task2_submission.csv')
