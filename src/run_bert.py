import torch
import pandas as pd
import pickle
import os
from argparse import ArgumentParser
from dataset import BertDataset
import ipdb
from utils import plot_word_analysis


def main():
    parser = ArgumentParser()
    parser.add_argument("--do_data", action='store_true')
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument('--model', default='bert-base-cased', type=str, help='pretrained_model_name')
    parser.add_argument('--max_len', default=512, type=int)
    parser.add_argument('--epochs', default=6, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--accum', default=1, type=int, help='gradient_accumulation_steps')
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--cuda', default=-1, type=int)
    parser.add_argument('--checkpoint', default=-1, type=int)
    parser.add_argument('--fz', default=-1, type=int)
    parser.add_argument('--lr_step', default=10, type=int, help='learning rate scheduler step size')
    parser.add_argument('--gamma', default=0.5, type=float, help='learning rate scheduler gamma')
    parser.add_argument('--load_model', action='store_true')
    args = parser.parse_args()

    global data_name
    data_name = '%s_%d' % (args.model.split('-', 1)[1], args.max_len)
    
    global dir_name
    dir_name = '0_%s_%d_lr%.1E' % (args.model.split('-', 1)[1], args.max_len, args.lr)
    if not args.accum == 1:
        dir_name += '_accum%d' % args.accum
    if not args.fz == -1:
        dir_name += '_fz%d' % args.fz
    if not args.lr_step == 10:
        dir_name += '_lr-step%d' % args.lr_step
    dir_name += '_NOaug'
    print('dir name: %s' % dir_name)

    if args.do_data:
        preprocess(args, data_name)

    if args.do_train:
        if not os.path.exists('../dataset/trainData_%s.pkl' % data_name):
            preprocess(args, data_name)
        train(args, data_name)

    if args.do_test:
        predict(args, data_name)


def preprocess(args, data_name):
    from utils import remove_info
    from sklearn.model_selection import train_test_split
    from preprocessor import Preprocessor
    torch.manual_seed(42)

    print('[Info] Process csv...')
    # for train and valid csv
    trainset = pd.read_csv('../data/task2_trainset.csv', dtype=str)
    trainset = remove_info(trainset)
    trainset, validset = train_test_split(trainset, test_size=0.2, random_state=42)
    testset = pd.read_csv('../data/task2_public_testset.csv', dtype=str)
    testset = remove_info(testset)

    print('[INFO] Make bert dataset...')
    preprocessor = Preprocessor(args.model)
    train_data = preprocessor.get_dataset(trainset, n_workers=12)
    valid_data = preprocessor.get_dataset(validset, n_workers=12)
    test_data = preprocessor.get_dataset(testset, n_workers=12)

    plot_word_analysis(train_data + valid_data + test_data)

    train_data = BertDataset(train_data, args.max_len, test=False)
    valid_data = BertDataset(valid_data, args.max_len, test=True)
    test_data = BertDataset(test_data, args.max_len, test=True)

    print('[INFO] Save pickles...')
    if not os.path.exists('../dataset/'):
        os.makedirs('../dataset/')
    with open('../dataset/trainData_%s.pkl' % data_name, 'wb') as f:
        pickle.dump(train_data, f)
    with open('../dataset/validData_%s.pkl' % data_name, 'wb') as f:
        pickle.dump(valid_data, f)
    with open('../dataset/testData_%s.pkl' % data_name, 'wb') as f:
        pickle.dump(test_data, f)


def train(args, data_name):
    from trainer import Trainer
    from network import BertForMultiLabelSequenceClassification
    from utils import plot

    with open('../dataset/trainData_%s.pkl' % data_name, 'rb') as f:
        train_data = pickle.load(f)
    with open('../dataset/validData_%s.pkl' % data_name, 'rb') as f:
        valid_data = pickle.load(f)

    device = torch.device('cuda:%d' % args.cuda if torch.cuda.is_available() else 'cpu')
    model = BertForMultiLabelSequenceClassification.from_pretrained(args.model, num_labels=4)
    if args.load_model:
        model.load_state_dict(torch.load('../model/%s/model.pkl.%d' % (dir_name, args.checkpoint)))
    model.to(device)

    trainer = Trainer(device, model, args.batch_size, args.lr, args.accum, args.grad_clip, args.fz,
                      args.lr_step, args.gamma)

    for epoch in range(args.epochs):
        print('dir name: %s' % dir_name)
        print('Epoch: {}'.format(epoch))
        trainer.run_epoch(epoch, train_data, True)
        trainer.run_epoch(epoch, valid_data, False)
        trainer.save(epoch, dir_name)
    plot(dir_name)


def predict(args, data_name):
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from network import BertForMultiLabelSequenceClassification
    from utils import SubmitGenerator

    with open('../dataset/testData_%s.pkl' % data_name, 'rb') as f:
        testData = pickle.load(f)

    device = torch.device('cuda:%d' % args.cuda if torch.cuda.is_available() else 'cpu')
    model = BertForMultiLabelSequenceClassification.from_pretrained(args.model, num_labels=4)
    model.load_state_dict(torch.load('../model/%s/model.pkl.%d' % (dir_name, args.checkpoint)))
    model.train(False)
    model.to(device)

    dataloader = DataLoader(dataset=testData,
                            batch_size=args.batch_size,
                            shuffle=False,
                            collate_fn=testData.collate_fn,
                            num_workers=1)
    trange = tqdm(enumerate(dataloader), total=len(dataloader), desc='Predict')
    prediction = []
    for i, (tokens, segments, masks, labels) in trange:
        with torch.no_grad():
            o_labels = model(tokens.to(device), segments.to(device), masks.to(device))
            o_labels = o_labels > 0.0
            prediction.append(o_labels.to('cpu'))

    prediction = torch.cat(prediction).detach().numpy().astype(int)

    if not os.path.exists('../score/task 2/'):
        os.makedirs('../score/task 2/')
    SubmitGenerator(prediction, '../data/task2_sample_submission.csv', True,
                    '../score/task 2/task2_submission_320.csv')


if __name__ == '__main__':
    main()
