import argparse
import json
import os
import pickle as pkl
import pprint
import numpy as np
import torch
import torch.utils.data as data
import torchnet as tnt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from dataset import PixelSetDataDifferences
from learning.focal_loss import FocalLoss
from learning.metrics import mIou, confusion_matrix_analysis
from learning.weight_init import weight_init
from models.pse_tcn_tae import PseTCNTae


def train_epoch(model, optimizer, criterion, data_loader, device, config):
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
    loss_meter = tnt.meter.AverageValueMeter()
    y_true = []
    y_pred = []
    for i, (x, diff, y) in enumerate(data_loader):
        y_true.extend(list(map(int, y)))
        x = recursive_todevice(x, device)
        diff = recursive_todevice(diff, device)
        y = y.to(device)
        optimizer.zero_grad()
        out = model(x, diff)
        loss = criterion(out, y.long())
        # torch.nn.utils.clip_grad_value_(model.parameters(), 1.)
        loss.backward()
        optimizer.step()
        pred = out.detach()
        y_p = pred.argmax(dim=1).cpu().numpy()
        y_pred.extend(list(y_p))
        acc_meter.add(pred, y)
        loss_meter.add(loss.item())
        if (i + 1) % config['display_step'] == 0:
            print('Step [{}/{}], Loss: {:.4f}, Acc : {:.2f}'
                  .format(i + 1, len(data_loader),
                    loss_meter.value()[0],
                    acc_meter.value()[0],
                  ))
    epoch_metrics = {'train_loss': loss_meter.value()[0],
                     'train_accuracy': acc_meter.value()[0],
                     'train_IoU': mIou(y_true, y_pred, n_classes=config['num_classes'])
                     }
    return epoch_metrics


def evaluation(model, criterion, loader, device, config, mode='val'):
    y_true = []
    y_pred = []
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
    loss_meter = tnt.meter.AverageValueMeter()

    for (x, diff, y) in loader:
        y_true.extend(list(map(int, y)))
        x = recursive_todevice(x, device)
        diff = recursive_todevice(diff, device)
        y = y.to(device)
        with torch.no_grad():
            prediction = model(x, diff)
            loss = criterion(prediction, y)
        acc_meter.add(prediction, y)
        loss_meter.add(loss.item())
        y_p = prediction.argmax(dim=1).cpu().numpy()
        y_pred.extend(list(y_p))
    metrics = {'{}_accuracy'.format(mode): acc_meter.value()[0],
               '{}_loss'.format(mode): loss_meter.value()[0],
               '{}_IoU'.format(mode): mIou(y_true, y_pred, config['num_classes']),
               }
    if mode == 'val':
        return metrics
    elif mode == 'test':
        return metrics, confusion_matrix(y_true, y_pred, labels=list(range(config['num_classes'])))


def get_loaders(dt, kfold, config):
    indices = list(range(len(dt)))
    np.random.shuffle(indices)
    kf = KFold(n_splits=kfold, shuffle=False)
    indices_seq = list(kf.split(list(range(len(dt)))))
    ntest = len(indices_seq[0][1])
    loader_seq = []
    for trainval, test_indices in indices_seq:
        trainval = [indices[i] for i in trainval]
        test_indices = [indices[i] for i in test_indices]
        validation_indices = trainval[-ntest:]
        train_indices = trainval[:-ntest]
        train_sampler = data.sampler.SubsetRandomSampler(train_indices)
        validation_sampler = data.sampler.SubsetRandomSampler(validation_indices)
        test_sampler = data.sampler.SubsetRandomSampler(test_indices)
        train_loader = data.DataLoader(dt, batch_size=config['batch_size'],
                                       sampler=train_sampler,
                                       num_workers=config['num_workers'])
        validation_loader = data.DataLoader(dt, batch_size=config['batch_size'],
                                            sampler=validation_sampler,
                                            num_workers=config['num_workers'])
        test_loader = data.DataLoader(dt, batch_size=config['batch_size'],
                                      sampler=test_sampler,
                                      num_workers=config['num_workers'])

        loader_seq.append((train_loader, validation_loader, test_loader))
    return loader_seq


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return [recursive_todevice(c, device) for c in x]


def prepare_output(config):
    os.makedirs(config['res_dir'], exist_ok=True)
    for fold in range(1, config['kfold'] + 1):
        os.makedirs(os.path.join(config['res_dir'], 'Fold_{}'.format(fold)), exist_ok=True)


def checkpoint(fold, log, config):
    with open(os.path.join(config['res_dir'], 'Fold_{}'.format(fold), 'trainlog.json'), 'w') as outfile:
        json.dump(log, outfile, indent=4)


def save_results(fold, metrics, conf_mat, config):
    with open(os.path.join(config['res_dir'], 'Fold_{}'.format(fold), 'test_metrics.json'), 'w') as outfile:
        json.dump(metrics, outfile, indent=4)
    pkl.dump(conf_mat, open(os.path.join(config['res_dir'], 'Fold_{}'.format(fold), 'conf_mat.pkl'), 'wb'))


def overall_performance(config):
    cm = np.zeros((config['num_classes'], config['num_classes']))
    for fold in range(1, config['kfold'] + 1):
        cm += pkl.load(open(os.path.join(config['res_dir'], 'Fold_{}'.format(fold), 'conf_mat.pkl'), 'rb'))
    _, perf = confusion_matrix_analysis(cm)
    print('Overall performance:')
    print('Acc: {},  IoU: {}'.format(perf['Accuracy'], perf['MACRO_IoU']))
    with open(os.path.join(config['res_dir'], 'overall.json'), 'w') as file:
        file.write(json.dumps(perf, indent=4))


def main(config):
    np.random.seed(config['rdm_seed'])
    torch.manual_seed(config['rdm_seed'])
    prepare_output(config)
    print('pytorch version: ', torch.__version__)
    print('cuda version: ', torch.version.cuda)
    mean_std = pkl.load(open(config['dataset_folder'] + '/S2-2017-T31TFM-meanstd.pkl', 'rb'))
    extra = 'geomfeat' if config['geomfeat'] else None
    dt = PixelSetDataDifferences(config['dataset_folder'], labels='label_44class', npixel=config['npixel'],
                      sub_classes=[1, 3, 4, 5, 6, 8, 9, 12, 13, 14, 16, 18, 19, 23, 28, 31, 33, 34, 36, 39],
                      norm=mean_std,
                      extra_feature=extra)
    device = torch.device(config['device'] + ':0')
    loaders = get_loaders(dt, config['kfold'], config)

    with open(os.path.join(config['res_dir'], 'training_arguments.txt'), 'w') as f:
        json.dump(config, f, indent=2)

    for fold, (train_loader, val_loader, test_loader) in enumerate(loaders):
        print('Starting Fold {}'.format(fold + 1))
        print('Train {}, Val {}, Test {}'.format(len(train_loader), len(val_loader), len(test_loader)))

        model_config = dict(input_dim=config['input_dim'], mlp1=config['mlp1'], pooling=config['pooling'],
                            mlp2=config['mlp2'], n_head=config['n_head'], d_k=config['d_k'], mlp3=config['mlp3'],
                            dropout=config['dropout'], T=config['T'], len_max_seq=config['lms'],
                            mlp4=config['mlp4'], d_model=config['d_model'], positions=dt.date_positions)
        if config['geomfeat']:
            model_config.update(with_extra=True, extra_size=4)
        else:
            model_config.update(with_extra=False, extra_size=None)

        model = PseTCNTae(**model_config)
        print(model.param_ratio())
        model = model.to(device)
        model.apply(weight_init)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        criterion = FocalLoss(config['gamma'])
        trainlog = {}
        best_mIoU = 0
        for epoch in range(1, config['epochs'] + 1):
            print('EPOCH {}/{}'.format(epoch, config['epochs']))

            model.train()
            train_metrics = train_epoch(model, optimizer, criterion, train_loader, device=device, config=config)

            print('Validation . . . ')
            model.eval()
            val_metrics = evaluation(model, criterion, val_loader, device=device, config=config, mode='val')

            print('Loss {:.4f},  Acc {:.2f},  IoU {:.4f}'
                .format(
                val_metrics['val_loss'],
                val_metrics['val_accuracy'],
                val_metrics['val_IoU'],
            ))

            trainlog[epoch] = {**train_metrics, **val_metrics}
            checkpoint(fold + 1, trainlog, config)

            if val_metrics['val_IoU'] >= best_mIoU:
                best_mIoU = val_metrics['val_IoU']
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           os.path.join(config['res_dir'], 'Fold_{}'.format(fold + 1), 'model.pth.tar'))

        print('Testing best epoch . . .')
        model.load_state_dict(
            torch.load(os.path.join(config['res_dir'], 'Fold_{}'.format(fold + 1), 'model.pth.tar'))['state_dict'])
        model.eval()

        test_metrics, conf_mat = evaluation(model, criterion, test_loader, device=device, mode='test', config=config)

        print('Loss {:.4f},  Acc {:.2f},  IoU {:.4f}'.format(test_metrics['test_loss'], test_metrics['test_accuracy'],
                                                             test_metrics['test_IoU']))
        save_results(fold + 1, test_metrics, conf_mat, config)

    overall_performance(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--dataset_folder', default='/home/andrster/workplace/datasets/french_satelite_dataset/S2-2017-T31TFM-PixelSet', type=str,
                        help='Path to the folder where the results are saved.')
    parser.add_argument('--res_dir', default='results/tst', help='Path to the folder where the results should be stored')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers')
    parser.add_argument('--rdm_seed', default=1, type=int, help='Random seed')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--display_step', default=50, type=int,
                        help='Interval in batches between display of training metrics')
    parser.set_defaults(preload=False)
    # Training parameters
    parser.add_argument('--kfold', default=5, type=int, help='Number of folds for cross validation')
    parser.add_argument('--epochs', default=300, type=int, help='Number of epochs per fold')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--gamma', default=1, type=float, help='Gamma parameter of the focal loss')
    parser.add_argument('--npixel', default=64, type=int, help='Number of pixels to sample from the input images')
    # Architecture Hyperparameters
    ## PSE
    parser.add_argument('--input_dim', default=10, type=int, help='Number of channels of input images')
    parser.add_argument('--mlp1', default='[10,32,64]', type=str, help='Number of neurons in the layers of MLP1')
    parser.add_argument('--pooling', default='mean_std', type=str, help='Pixel-embeddings pooling strategy')
    parser.add_argument('--mlp2', default='[132,128]', type=str, help='Number of neurons in the layers of MLP2')#[132,256] for LTAE
    parser.add_argument('--geomfeat', default=1, type=int,
                        help='If 1 the precomputed geometrical features (f) are used in the PSE.')
    ## TCN-TAE
    parser.add_argument('--n_head', default=16, type=int, help='Number of attention heads')#4
    parser.add_argument('--d_k', default=8, type=int, help='Dimension of the key and query vectors')#32
    parser.add_argument('--mlp3', default='[384,256,128]', type=str, help='Number of neurons in the layers of MLP3')#[512,128,128]
    parser.add_argument('--T', default=1000, type=int, help='Maximum period for the positional encoding')
    parser.add_argument('--lms', default=24, type=int,
                        help='Maximum sequence length for positional encoding (only necessary if positions == order)')
    parser.add_argument('--dropout', default=0.2, type=float, help='Dropout probability')
    parser.add_argument('--d_model', default=256, type=int,
                        help="size of the embeddings (E), if input vectors are of a different size,"
                             " a linear layer is used to project them to a d_model-dimensional space")
    ## Classifier
    parser.add_argument('--num_classes', default=20, type=int, help='Number of classes')
    parser.add_argument('--mlp4', default='[768, 384, 256,128,64,32,20]', type=str, help='Number of neurons in the layers of MLP4')
    config = parser.parse_args()
    config = vars(config)
    for k, v in config.items():
        if 'mlp' in k:
            v = v.replace('[', '')
            v = v.replace(']', '')
            config[k] = list(map(int, v.split(',')))
    pprint.pprint(config)
    main(config)