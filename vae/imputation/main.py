
import torch
import numpy as np
import pandas as pd
from models.modules import PointNet, MaskNet, DenseDecoder
from networks.enc import DenseGaussianNet
from models import PartialVAE, notMIWAE, MNARPartialVAE

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#print(f'using device: {device}')

def impute_by_PVAE(x, device):
    torch.manual_seed(0)
    ptnet = PointNet(x.shape[1], 16, 16, None, device, True, 'sum')
    enc = DenseGaussianNet([16, 8]).to(device)
    dec = DenseGaussianNet([8, 16, x.shape[1]]).to(device)
    opt = torch.optim.Adam(lr=1e-3, params=list(ptnet.parameters()) + list(enc.parameters()) + list(dec.parameters()),
                           weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.1)

    pvae = PartialVAE(
        ptnet, enc, dec, opt, sch, {'batch_size': 64, 'max_epochs': 300, 'device': device}
    )

    pvae.fit(x)
    return pvae.reconstruct(x)

def impute_by_notMIWAE(x, device):
    torch.manual_seed(0)
    ptnet = PointNet(x.shape[1], 16, 16, None, device, True, 'sum')
    maskdec = DenseDecoder(x.shape[1], x.shape[1], device)
    enc = DenseGaussianNet([16, 8]).to(device)
    dec = DenseGaussianNet([8, 16, x.shape[1]]).to(device)
    opt = torch.optim.Adam(lr=1e-3, params=list(ptnet.parameters()) + list(enc.parameters()) + list(dec.parameters()),
                           weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.1)
    notmiwae = notMIWAE(ptnet, maskdec, enc, dec, opt, sch, {'batch_size': 64, 'max_epochs': 300, 'device': device})
    notmiwae.fit(x)
    return notmiwae.reconstruct(x)


def impute_by_GINA(x, device):
    torch.manual_seed(0)
    masknet = MaskNet(x.shape[1], device)
    ptnet = PointNet(x.shape[1], 16, 16, None, device, True, 'sum')
    enc = DenseGaussianNet([16, 8]).to(device)
    dec = DenseGaussianNet([8, 16, x.shape[1]]).to(device)
    opt = torch.optim.Adam(lr=1e-3, params=list(ptnet.parameters()) + list(masknet.parameters()) + list(enc.parameters()) + list(dec.parameters()),
                           weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=150, gamma=0.1)

    gina = MNARPartialVAE(ptnet, masknet, enc, dec, opt, sch, {'batch_size': 64, 'max_epochs': 300, 'device': device})
    gina.fit(x)
    return gina.reconstruct(x)

if __name__ == '__main__':
    import os, argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, type=str, help='CSV file that contains data with missing values. Do not include missing value indicators.')
    parser.add_argument('--method', required=True, type=str, help='method to impute the missing values: one of ["pvae", "notmiwae", "gina"]')
    parser.add_argument('--save_dir', default='result', type=str)
    parser.add_argument('--gpu',  action='store_false', default=False)
    args = parser.parse_args()

    device = 'cuda' if args.gpu else 'cpu'
    assert args.method in ['pvae', 'notmiwae', 'gina']
    assert os.path.isfile(args.input_file), f'{args.input_file} does not exist'
    path = Path(args.save_dir)
    path.mkdir(parents=True, exist_ok=True)

    raw_data = pd.read_csv(args.input_file, index_col=0)
    target_cols = [c for c in raw_data.columns if 'obs' in c]
    data = raw_data[target_cols]
    x = data.values
    m = np.isnan(x)

    if args.method == 'pvae':
        impute_fn = impute_by_PVAE
    elif args.method == 'notmiwae':
        impute_fn = impute_by_notMIWAE
    elif args.method == 'gina':
        impute_fn = impute_by_GINA

    x_imputed = impute_fn(x, device)
    x[m] = x_imputed[m]

    out_file = os.path.basename(args.input_file).replace('.csv', '')
    raw_data[target_cols] = x
    #out = pd.DataFrame(x, index=data.index, columns=data.columns)
    raw_data.to_csv(os.path.join(args.save_dir, f'{out_file}_{args.method}_imputed.csv'), index=None)



