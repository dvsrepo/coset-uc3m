from torchtext import data
import spacy
import visdom
import numpy as np
import torch
from torch import nn

from sklearn.metrics import f1_score

from model import RNNClassifier

from sklearn.utils import class_weight

import utils

vis = visdom.Visdom()


def train(model, batches, num_epochs=2, lot=None, optimizer=None, lr=0.0001, lr_decay=0.9, class_weights = None):
    import time
    train_iter, dev_iter = batches
    criterion = nn.CrossEntropyLoss()

    from torch.autograd import Variable
    # Now the code for training our network
    iterations = 0
    start = time.time()
    for epoch in range(num_epochs):
        model.train();
        train_iter.init_epoch()
        n_correct, n_total, f1, total_loss = 0, 0, 0, 0
        predictions, gold_labels = [], []
        for batch_idx, batch in enumerate(train_iter):
            optimizer.zero_grad()
            output = model(batch.text)
            iterations += 1
            n_correct += (torch.max(output, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
            n_total += batch.batch_size
            predictions.extend(torch.max(output, 1)[1].view(batch.label.size()).data.numpy())
            gold_labels.extend(batch.label.data.numpy())
            loss = criterion(output, batch.label)
            total_loss += loss
            loss.backward()
            optimizer.step()

        train_acc = f1_score(predictions, gold_labels, average='macro')

        model.eval(); dev_iter.init_epoch()
        n_dev_correct, n_dev_total, f1, dev_loss = 0, 0, 0, 0
        predictions, gold_labels = [], []
        for dev_batch_idx, dev_batch in enumerate(dev_iter):
            answer = model(dev_batch.text)
            n_dev_correct += (torch.max(answer, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
            predictions.extend(torch.max(answer, 1)[1].view(dev_batch.label.size()).data.numpy())
            gold_labels.extend(dev_batch.label.data.numpy())
            n_dev_total += dev_batch.batch_size
            dev_loss += criterion(answer, dev_batch.label)

        dev_acc = f1_score(predictions, gold_labels, average='macro')
        utils.log(time.time()-start,
                        epoch,
                        iterations,
                        batch_idx,
                        train_iter,
                        total_loss.data[0]/len(train_iter),
                        train_acc,
                        dev_loss.data[0]/len(dev_iter),
                        dev_acc,
                        lot=lot,
                        vis=vis)
        utils.adjust_learning_rate(lr, lr_decay, optimizer, epoch, 1)



def trainer(batch_size, lr, min_freq=None, vocab_size=None, model=None, optimizer=None, lr_decay=0.98, class_weights = None):
    print('Starting training with batch size {}'.format(batch_size))
    train_iter, dev_iter = data.BucketIterator.splits((trainset, devset),
                                                  batch_size=batch_size,
                                                  sort_key=lambda x: len(x.text),
                                                  device=-1,
                                                  repeat=False)
    lot = vis.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 2)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Loss',
                title='LOSS - Batch.{}.LR.{}.Vocab_size.{}.Min_freq.{}'.format(batch_size,lr, vocab_size, min_freq),
                legend=['Train Loss', 'Dev Loss']
            )
        )
    lot_acc = vis.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 2)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='F1-Macro',
                title='F1-Macro - Batch.{}.LR.{}.Vocab_size.{}.Min_freq.{}'.format(batch_size,lr, vocab_size, min_freq),
                legend=['Train F1', 'Dev F1']
            )
        )

    train(model, (train_iter, dev_iter),num_epochs=10, lot=[lot, lot_acc], optimizer=optimizer, lr=lr, lr_decay=lr_decay, class_weights=class_weights)

PATH_TO_DATA = 'data/'
batch_sizes = [10]
hidden_size = 300
rnn_hidden_size = 150
embedding_size = 300
INITIALIZE_EMBEDDINGS = 'fasttext'
BALANCED_CLASS_WEIGHTS = False
min_frequency = 2
TEXT, LABEL, trainset, devset, _ = utils.get_datasets(PATH_TO_DATA)

lrs = [0.0005]

TEXT.build_vocab(trainset, devset, min_freq=min_frequency, wv_dir=PATH_TO_DATA, wv_type='fasttext', wv_dim=300, unk_init=None)

model = RNNClassifier(len(TEXT.vocab), hidden_size,
                      rnn_hidden_size, len(LABEL.vocab),
                      num_layers=5, dropout=0.4)

model = utils.initialize_embedding_layer(model, TEXT.vocab,
                                         embedding_size, INITIALIZE_EMBEDDINGS,
                                         fixed_embeddings = False)
print(model)
optimizer = torch.optim.Adam( filter(lambda p: p.requires_grad, model.parameters())) # We could try to set weight_decay=1e-6 for regularization, weight_decay=1e-3

class_weights = utils.get_class_weights(LABEL.vocab, trainset, BALANCED_CLASS_WEIGHTS)

for batch_size in batch_sizes:
    for lr in lrs:
        trainer(batch_size, lr, min_freq=min_frequency, vocab_size=len(TEXT.vocab), model=model, optimizer=optimizer, lr_decay=.99, class_weights = class_weights )
