from utils import *
import tqdm
acc_all, result = [], []
import torch


def train_auxi(args, source_loaders, DEVICE, model, optimizer, e, false_optimizer):
    model.train()
    train_loss = 0
    epoch_class_y_loss = 0
    total = 0
    for source_loader in source_loaders:
        for batch_idx, (x, y, d) in enumerate(source_loader):
            x, y, d = x.to(DEVICE), y.to(DEVICE), d.to(DEVICE)

            optimizer.zero_grad()
            false_optimizer.zero_grad()

            loss_origin, class_y_loss = model.loss_function(d, x, y)
            loss_origin = loss_origin

            loss_false = model.loss_function_false(args, d, x, y)

            loss_origin.backward()
            optimizer.step()
            loss_false.backward()
            false_optimizer.step()

            train_loss += loss_origin
            epoch_class_y_loss += class_y_loss
            total += y.size(0)

    train_loss /= total
    epoch_class_y_loss /= total

    return train_loss, epoch_class_y_loss


def get_accuracy(source_loaders, DEVICE, model, classifier_fn, batch_size):
    model.eval()
    """
    compute the accuracy over the supervised training set or the testing set
    """
    predictions_d, actuals_d, predictions_y, actuals_y = [], [], [], []
    predictions_d_false, predictions_y_false = [], []

    with torch.no_grad():
        for source_loader in source_loaders:
            for (xs, ys, ds) in source_loader:

                xs, ys, ds = xs.to(DEVICE), ys.to(DEVICE), ds.to(DEVICE)

                # use classification function to compute all predictions for each batch
                pred_d, pred_y, pred_d_false, pred_y_false = classifier_fn(xs)
                predictions_d.append(pred_d)
                predictions_d_false.append(pred_d_false)
                actuals_d.append(ds)

                predictions_y.append(pred_y)
                predictions_y_false.append(pred_y_false)
                actuals_y.append(ys)

        # compute the number of accurate predictions
        accurate_preds_d = 0
        accurate_preds_d_false = 0
        for pred, act in zip(predictions_d, actuals_d):
            _, num_pred = pred.max(1)
            v = torch.sum(num_pred == act)
            accurate_preds_d += v

        accuracy_d = (accurate_preds_d * 1.0) / (len(predictions_d) * batch_size)

        for pred, act in zip(predictions_d_false, actuals_d):
            _, num_pred = pred.max(1)
            v = torch.sum(num_pred == act)
            accurate_preds_d_false += v

        # calculate the accuracy between 0 and 1
        accuracy_d_false = (accurate_preds_d_false * 1.0) / (len(predictions_d_false) * batch_size)

        # compute the number of accurate predictions
        accurate_preds_y = 0
        accurate_preds_y_false = 0

        for pred, act in zip(predictions_y, actuals_y):
            _, num_pred = pred.max(1)
            v = torch.sum(num_pred == act)
            accurate_preds_y += v

        accuracy_y = (accurate_preds_y * 1.0) / (len(predictions_y) * batch_size)


        for pred, act in zip(predictions_y_false, actuals_y):
            _, num_pred = pred.max(1)
            v = torch.sum(num_pred == act)
            accurate_preds_y_false += v
        # calculate the accuracy between 0 and 1
        accuracy_y_false = (accurate_preds_y_false * 1.0) / (len(predictions_y_false) * batch_size)

        return accuracy_d, accuracy_y, accuracy_d_false, accuracy_y_false



def train_GILE(model, DEVICE, optimizer, source_loaders, target_loader, result_csv, result_txt, args):
    acc_all, result = [], []
    best_acc, best_iter = 0.0, 0

    false_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for e in range(args.n_epoch):
        avg_epoch_loss, avg_epoch_class_y_loss = train_auxi(args, source_loaders, DEVICE, model, optimizer, e, false_optimizer)
        tqdm.tqdm.write('Epoch: [{}/{}], avg loss: {:.2f}, y loss: {:.2f}%'.format(e + 1, args.n_epoch, avg_epoch_loss, avg_epoch_class_y_loss))
        train_acc_d, train_acc_y, train_acc_d_false, train_acc_y_false = get_accuracy(source_loaders, DEVICE, model, model.classifier, args.batch_size)
        test_acc_d, test_acc_y, test_acc_d_false, test_acc_y_false = get_accuracy([target_loader], DEVICE, model, model.classifier, args.batch_size)
        tqdm.tqdm.write('Epoch:[{}/{}], train acc d:{:.2f} y:{:.2f} | d_false:{:.2f} y_false:{:.2f}'.format(e+1, args.n_epoch, train_acc_d, train_acc_y, train_acc_d_false, train_acc_y_false))
        tqdm.tqdm.write('Epoch:[{}/{}], TEST acc d:{:.2f} y:{:.2f} | d_false:{:.2f} y_false:{:.2f}'.format(e+1, args.n_epoch, test_acc_d, test_acc_y, test_acc_d_false, test_acc_y_false))

        acc_all.append(test_acc_y)
        best_acc = max(acc_all).item()
        best_iter = acc_all.index(best_acc) + 1
        result.append([train_acc_y, test_acc_y, best_acc, best_iter, train_acc_d])
        result_np = np.array(result, dtype=float)
        np.savetxt(result_csv, result_np, fmt='%.2f', delimiter=',')
    log_result_concise(args, result_txt, [best_acc, best_iter])
    plot(result_csv)


def train(model, DEVICE, optimizer, source_loaders, target_loader, result_csv, result_txt, args):
    if args.now_model_name in ['GILE']:
        return train_GILE(model, DEVICE, optimizer, source_loaders, target_loader, result_csv, result_txt, args)
    else:
        pass