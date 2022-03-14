import multiprocessing
import sys
import os
import pickle
from multiprocessing import Process
import logging

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess


import lib.print
import lib.dump


def run_bash(cmd, cwd=None):
    logging.info("running" + str(cmd))
    if cwd:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, cwd=cwd)
    else:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    output, error = process.communicate()
    logging.info(output)


def run_banshap_binaries(BST_NAME, DATA_NAME):
    #run_bash(["make"], "build")
    processes = []
    for s in ['f', 'g', 'h', 'o', 'i']:
        arg = ["time", "../build/shap_banzhaf", s, f"{BST_NAME}", f"{DATA_NAME}"]
        processes.append(Process(target=run_bash, args=(arg,)))

    for p in processes:
        p.start()

    for p in processes:
        p.join()



def compute_error_plots(feats_importances, NAME):
    a=feats_importances['shap_simple']
    b=feats_importances['banzhaf_fast']

    lib.print.plot_vals_wide((a-b).abs().mean().to_frame().transpose(), f'{NAME}_l1', dir='errors')
    plt.show()

    plot = lib.print.plot_vals_wide(((a-b)*(a-b)).mean().apply(lambda x:np.sqrt(x)).to_frame().transpose(), f'{NAME}_l2', dir='errors')
    plt.show()

    plot = lib.print.plot_vals_wide(((a-b)/b).fillna(0).abs().mean().to_frame().transpose(), f'{NAME}_mape', dir='errors')
    plt.show()


def compute_shap_orig(SHAP_ORIG_NAME, X, model):
    logging.info('Start compute shap orig')
    explainer = shap.TreeExplainer(model)
    #return
    shap_values = explainer.shap_values(X, check_additivity=False)
    #random forst classifier returns values for both classes
    if type(shap_values)==list:
        shap_values = shap_values[0]
    shap_original = pd.DataFrame(shap_values, columns=X.columns)
    logging.info('Finish compute shap orig')

    shap_original.to_csv(SHAP_ORIG_NAME, index=False)


def compute_instance(NAME, X, model, model_type):
    logging.basicConfig(filename=f'{NAME}.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)
    logging.info('COMPUTE_INSTANCE '+ NAME)

    DATA_PATH=f'../model/{NAME}/'
    BST_NAME=f'{DATA_PATH}/bst_{NAME}.file'
    DATA_NAME=f'{DATA_PATH}/{NAME}.csv'
    SHAP_ORIG_NAME=f'{BST_NAME}.shap_orig'
    run_bash(['mkdir', '-p', f'{DATA_PATH}'])


    #saving tree and model
    lib.dump.dump_trees(model, BST_NAME, X.columns, model_type)
    os.makedirs("binary_models", exist_ok=True)
    pickle.dump(model, open(f"binary_models/{NAME}.pickle.dat", "wb"))
    pd.DataFrame(X).to_csv(DATA_NAME, index=False)


    # we skip this, there is strange error for flights_DT_100 instance
    #native_shap_process = Process(target=compute_shap_orig, args=(SHAP_ORIG_NAME, X, model))
    #native_shap_process.start()

    run_banshap_binaries(BST_NAME, DATA_NAME)
    #native_shap_process.join()

    #tags = ['shap_simple', 'shap_fast', 'banzhaf_simple', 'banzhaf_fast', 'shap_orig_c', 'shap_orig']
    tags = ['shap_simple', 'shap_fast', 'banzhaf_simple', 'banzhaf_fast', 'shap_orig_c' ]
    feats_importances = {}

    for t in tags:
        feats_importances[t] = pd.read_csv(f'{BST_NAME}.{t}')



    compute_error_plots(feats_importances, NAME)

    #a = feats_importances['shap_orig']
    #b = feats_importances['shap_simple']
    #x= np.nan_to_num(np.abs((a - b) / b).values, nan=0, posinf=0, neginf=0)
    #np.mean(x)

    lib.print.compare_shaps(feats_importances)

    for limit in [1,3,10,20]:
        logging.info("limit = " + str(limit))
        trans_data, idx = lib.print.compare_transpositions(feats_importances, NAME, limit, plot=(limit==20))

    for t,df in feats_importances.items():
        logging.info(t)
        plot = lib.print.plot_vals(df, t, dir=f'importances/{NAME}')
        plt.show()

    logging.info("compute instance finished" + NAME)


