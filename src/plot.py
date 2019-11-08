def plot():
    import matplotlib.pyplot as plt
    import json
    import ipdb
    # ipdb.set_trace()

    with open('../model/0_base-cased_512_lr1.0E-05_NOaug/history.json', 'r') as f:
        history_0 = json.loads(f.read())

    train_loss_0 = [l['loss'] for l in history_0['train']]
    valid_loss_0 = [l['loss'] for l in history_0['valid']]
    train_f1_0 = [l['f1'] for l in history_0['train']]
    valid_f1_0 = [l['f1'] for l in history_0['valid']]

    plt.figure(figsize=(14, 7))
    plt.title('Loss')
    plt.plot(train_loss_0, label='bert train')
    plt.plot(valid_loss_0, '--', label='bert valid')
    plt.legend(loc='best')
    plt.savefig('../mix_loss.png')

    plt.figure(figsize=(14, 10))
    plt.title('F1 Score')
    plt.plot(valid_f1_0, label='bert valid')

    plt.legend(loc='best')
    plt.savefig('../mix_f1.png')

plot()
