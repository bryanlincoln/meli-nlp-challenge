import numpy as np
import pandas as pd
import torch
import copy
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from torch import nn
from ciclic_clr import CyclicLR
from model import Alex_NeuralNet_Meta, MyDataset
from utils import seed_everything


def pytorch_model_run_cv(x_train, y_train, y_train_lin, features, test_features, x_test, model_obj, params, feats=False, clip=True):
    seed_everything()
    avg_losses_f = []
    avg_val_losses_f = []

    x_test_cuda = torch.tensor(x_test, dtype=torch.long)
    test = torch.utils.data.TensorDataset(x_test_cuda)
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=params.batch_size, shuffle=False)

    splits = list(StratifiedKFold(n_splits=params.n_splits, shuffle=True,
                                  random_state=params.SEED).split(x_train, y_train_lin))
    for i, (train_idx, valid_idx) in enumerate(splits):
        seed_everything(i*1000+i)
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        if feats:
            features = np.array(features)
        x_train_fold = torch.tensor(
            x_train[train_idx.astype(int)], dtype=torch.long)
        y_train_fold = torch.tensor(
            y_train[train_idx.astype(int)], dtype=torch.float32)
        if feats:
            kfold_X_features = features[train_idx.astype(int)]
            kfold_X_valid_features = features[valid_idx.astype(int)]
        x_val_fold = torch.tensor(
            x_train[valid_idx.astype(int)], dtype=torch.long)
        y_val_fold = torch.tensor(
            y_train[valid_idx.astype(int)], dtype=torch.float32)

        model = copy.deepcopy(model_obj)

        model

        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')

        step_size = 300
        base_lr, max_lr = 0.001, 0.003
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=max_lr)

        scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
                             step_size=step_size, mode='exp_range',
                             gamma=0.99994)

        train = MyDataset(torch.utils.data.TensorDataset(
            x_train_fold, y_train_fold))
        valid = MyDataset(
            torch.utils.data.TensorDataset(x_val_fold, y_val_fold))

        train_loader = torch.utils.data.DataLoader(
            train, batch_size=params.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
            valid, batch_size=params.batch_size, shuffle=False)

        print(f'Fold {i + 1}')
        for epoch in range(params.n_epochs):
            start_time = time.time()
            model.train()

            avg_loss = 0.
            for i, (x_batch, y_batch, index) in enumerate(train_loader):
                if feats:
                    f = kfold_X_features[index]
                    y_pred = model([x_batch, f])
                else:
                    y_pred = model(x_batch)

                if scheduler:
                    scheduler.batch_step()

                # Compute and print loss.
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                if clip:
                    nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                avg_loss += loss.item() / len(train_loader)

            model.eval()
            # valid_preds_fold = np.zeros(x_val_fold.size(0))
            # test_preds_fold = np.zeros(len(x_test))

            avg_val_loss = 0.
            for i, (x_batch, y_batch, index) in enumerate(valid_loader):
                if feats:
                    f = kfold_X_valid_features[index]
                    y_pred = model([x_batch, f]).detach()
                else:
                    y_pred = model(x_batch).detach()

                avg_val_loss += loss_fn(y_pred,
                                        y_batch).item() / len(valid_loader)
                # valid_preds_fold[index] = y_pred.cpu().numpy()

            elapsed_time = time.time() - start_time
            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
                epoch + 1, params.n_epochs, avg_loss, avg_val_loss, elapsed_time))

        avg_losses_f.append(avg_loss)
        avg_val_losses_f.append(avg_val_loss)

    test_preds = False
    # predict all samples in the test set batch per batch
    for i, (x_batch,) in enumerate(test_loader):
        if feats:
            f = test_features[i *
                              params.batch_size:(i+1) * params.batch_size]
            y_pred = model([x_batch, f]).detach()
        else:
            y_pred = model(x_batch).detach()

        if test_preds is False:
            test_preds = y_pred.cpu().numpy()
        else:
            test_preds = np.append(test_preds, y_pred.cpu().numpy(), axis=0)

    print('All \t loss={:.4f} \t val_loss={:.4f} \t '.format(
        np.average(avg_losses_f), np.average(avg_val_losses_f)))
    return test_preds


def run(x_train, y_train, features, test_features, x_test, embedding_matrix, params):
    print('One-Hot encoding classes')
    ohe = OneHotEncoder()
    y_train_ec = ohe.fit_transform(y_train.reshape(-1, 1)).toarray()

    print('Running model')
    preds = pytorch_model_run_cv(x_train, y_train_ec, y_train, features, test_features, x_test, Alex_NeuralNet_Meta(
        70, 16, len(np.unique(y_train)), features, params, embedding_matrix), params, feats=True)

    print('Model trained, generating output...')

    if params.debug:
        df_test = pd.read_csv("../test.csv")[:200]
    else:
        df_test = pd.read_csv("../test.csv")

    df_test = df_test[df_test['language'] == (
        'portuguese' if params.lang == 'pt' else 'spanish')]
    submission = df_test[['id']].copy()
    submission['category'] = ohe.inverse_transform(preds)
    submission.to_csv('submission.csv', index=False)

    print('Output generated.')
