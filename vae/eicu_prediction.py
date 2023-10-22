
# this code requires imputed csv files in result directory

import warnings
warnings.filterwarnings('ignore')

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import math, numpy, sklearn.metrics.pairwise as sk
from cvxopt import matrix, solvers

def kmm(Xtrain, Xtest, sigma):
    n_tr = len(Xtrain)
    n_te = len(Xtest)

    # calculate Kernel
    print('Computing kernel for training data ...')
    K_ns = sk.rbf_kernel(Xtrain, Xtrain, sigma)
    # make it symmetric
    K = 0.9 * (K_ns + K_ns.transpose())

    # calculate kappa
    print('Computing kernel for kappa ...')
    kappa_r = sk.rbf_kernel(Xtrain, Xtest, sigma)
    ones = numpy.ones(shape=(n_te, 1))
    kappa = numpy.dot(kappa_r, ones)
    kappa = -(float(n_tr) / float(n_te)) * kappa

    # calculate eps
    eps = (math.sqrt(n_tr) - 1) / math.sqrt(n_tr)

    # constraints
    A0 = numpy.ones(shape=(1, n_tr))
    A1 = -numpy.ones(shape=(1, n_tr))
    A = numpy.vstack([A0, A1, -numpy.eye(n_tr), numpy.eye(n_tr)])
    b = numpy.array([[n_tr * (eps + 1), n_tr * (eps - 1)]])
    b = numpy.vstack([b.T, -numpy.zeros(shape=(n_tr, 1)), numpy.ones(shape=(n_tr, 1)) * 1000])

    print('Solving quadratic program for beta ...')
    P = matrix(K, tc='d')
    q = matrix(kappa, tc='d')
    G = matrix(A, tc='d')
    h = matrix(b, tc='d')
    beta = solvers.qp(P, q, G, h)
    return [i for i in beta['x']]

parser = argparse.ArgumentParser()
parser.add_argument('--source', default='Northeast', type=str)
parser.add_argument('--adapt', default=False, type=bool)
args = parser.parse_args()

regions = ['Northeast', 'South', 'Midwest', 'West']
assert args.source in regions
regions.remove(args.source)

dat = pd.read_csv('data/eicu.csv')
source_y = (dat[dat['region'] == args.source]['actualicumortality'] == 'EXPIRED').astype(float)
source_r = pd.read_csv(f'result/eicu_{args.source}_mean_1.csv', index_col=0)
missing_indicators = [c for c in source_r.columns if c.startswith('M_')]
source_r = source_r[missing_indicators].values

colnames = [c.replace('M_', '') for c in missing_indicators]

methods = ['mean', 'pvae', 'notmiwae', 'gina', 'mice_norm', 'mice_ri', 'mice_pmm', 'mice_rf']
for target in regions:
    target_y = (dat[dat['region'] == target]['actualicumortality'] == 'EXPIRED').astype(float)
    target_r = pd.read_csv(f'result/eicu_{target}_mean_1.csv', index_col=0)
    target_r = target_r[missing_indicators].values

    results = []
    pbar = tqdm(methods)
    for method in pbar:
        for seed in range(6):
            try:
                source_dat = pd.read_csv(f'result/eicu_{args.source}_{method}_{seed}.csv', index_col=0)
                target_dat = pd.read_csv(f'result/eicu_{target}_{method}_{seed}.csv', index_col=0)
            except:
                continue

            cat_colnames = ['ethnicity', 'gender', 'surgery']
            num_colnames = [c for c in colnames if c not in cat_colnames]

            if method.startswith('mice'):
                num_colnames = [c.replace(' ','.') for c in num_colnames]

            cat_encoder = OneHotEncoder(sparse=False, drop='first')
            num_scaler = StandardScaler()
            source_x_cat = cat_encoder.fit_transform(source_dat[cat_colnames])
            source_x_num = num_scaler.fit_transform(source_dat[num_colnames])

            target_x_cat = cat_encoder.transform(target_dat[cat_colnames])
            target_x_num = num_scaler.transform(target_dat[num_colnames])

            source_x = np.column_stack([source_x_num, source_x_cat, source_r])
            target_x = np.column_stack([target_x_num, target_x_cat, target_r])

            model = RandomForestRegressor(n_estimators=100)
            if args.adapt:
                weights = kmm(source_x, target_x, 0.1)
                model.fit(source_x, source_y, weights)
            else:
                model.fit(source_x, source_y)
            rmse = np.sqrt(mean_squared_error(target_y, model.predict(target_x)))

            results.append([method, seed, rmse])
            pbar.set_description(f'processing {args.source} - {target} - {method} - {seed:02d} - RMSE: {rmse:.4f}')

    if args.adapt:
        postfix = "adapted"
    else:
        postfix = "unadapted"

    results = pd.DataFrame(results, columns=['method', 'seed', 'RMSE'])
    results.to_csv(f'output/{args.source}-{target}-{postfix}.csv')

    sns.boxplot(data=results, x='RMSE', y='method', orient='h')
    plt.title(f'{args.source}-{target}.png')
    plt.savefig(f'output/{args.source}-{target}-{postfix}.png', dpi=400)
    plt.tight_layout()
    plt.close()
