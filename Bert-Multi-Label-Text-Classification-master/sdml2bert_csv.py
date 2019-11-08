import pandas as pd


def remove_info(dataset):
    dataset.drop('Title', axis=1, inplace=True)
    dataset.drop('Categories', axis=1, inplace=True)
    dataset.drop('Created Date', axis=1, inplace=True)
    dataset.drop('Authors', axis=1, inplace=True)
    return dataset


trainset = pd.read_csv('./pybert/sdml/raw_data/task2_trainset.csv', dtype=str)
trainset = remove_info(trainset)

labels = []
for i in trainset.iterrows():
    label = i[1]['Task 2'].split(' ')
    labels.append(
        [int('THEORETICAL' in label), int('ENGINEERING' in label),
         int('EMPIRICAL' in label), int('OTHERS' in label)]
    )
col_labels = [*zip(*labels)]

trainset.drop('Task 2', axis=1, inplace=True)
trainset.insert(2, "THEORETICAL", col_labels[0], True)
trainset.insert(3, "ENGINEERING", col_labels[1], True)
trainset.insert(4, "EMPIRICAL", col_labels[2], True)
trainset.insert(5, "OTHERS", col_labels[3], True)
trainset.to_csv('./pybert/sdml/sdml_train.csv', index=False)

testset = pd.read_csv('./pybert/sdml/raw_data/task2_public_testset.csv', dtype=str)
testset = remove_info(testset)
testset.to_csv('./pybert/sdml/sdml_test.csv', index=False)
