"""
credits to Vivien Sainte Fare Garnot (https://github.com/VSainteuf/pytorch-psetae/tree/master)
"""

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
from models.pse_tcn_tae import PseTCNTae


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


def get_loader(dt, kfold, config):
    indices = list(range(len(dt)))
    np.random.shuffle(indices)
    kf = KFold(n_splits=kfold, shuffle=False)
    indices_seq = list(kf.split(list(range(len(dt)))))
    loader_seq = []
    for _, test_indices in indices_seq:
        test_indices = [indices[i] for i in test_indices]
        test_sampler = data.sampler.SubsetRandomSampler(test_indices)

        test_loader = data.DataLoader(dt, batch_size=config['batch_size'],
                                      sampler=test_sampler,
                                      num_workers=config['num_workers'])

        loader_seq.append(test_loader)
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
    loader = get_loader(dt, config['kfold'], config)

    for fold, test_loader in enumerate(loader):
        print('Evaluating Fold {}'.format(fold + 1))

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
        criterion = FocalLoss(config['gamma'])

        model.load_state_dict(
            torch.load(os.path.join(config['pretrained_folder'], 'Fold_{}'.format(fold + 1), 'model.pth.tar'))['state_dict'])
        model.eval()

        test_metrics, conf_mat = evaluation(model, criterion, test_loader, device=device, mode='test', config=config)

        print('Loss {:.4f},  Acc {:.2f},  IoU {:.4f}'.format(test_metrics['test_loss'], test_metrics['test_accuracy'],
                                                             test_metrics['test_IoU']))
        save_results(fold + 1, test_metrics, conf_mat, config)

    overall_performance(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--dataset_folder', default='', type=str,
                        help='Path to the folder where the dataset is located.')
    parser.add_argument('--pretrained_folder', default='results/pretrained', type=str,
                        help='Path to pretrained model.')
    parser.add_argument('--res_dir', default='results/inference', help='Path to the folder where the results should be stored')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers')
    parser.add_argument('--rdm_seed', default=1, type=int, help='Random seed')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--display_step', default=50, type=int,
                        help='Interval in batches between display of training metrics')
    parser.set_defaults(preload=False)
    # Training parameters
    parser.add_argument('--kfold', default=5, type=int, help='Number of folds for cross validation')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
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