import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from random import sample,seed
from tqdm import tqdm
import pickle
import os

def plot_vals_wide(df, name=None, show=True, dir="."):
    os.makedirs(dir, exist_ok=True)
    val_pairs = list(sorted([(df[col].abs().mean(),col) for col in df.columns], reverse=True))[:20]
    vals = [x for x, y in val_pairs]
    names = [y for x, y in val_pairs]
    plot =  sns.barplot(vals, names, orient='h')
    if name is not None:
        plot.get_figure().savefig(f'{dir}/{name}.png', bbox_inches='tight')
    if show:
        plt.show()
    return plot

def plot_vals(df, name=None, show=True, dir="."):
    os.makedirs(dir, exist_ok=True)
    val_pairs = list(sorted([(df[col].abs().mean(),col) for col in df.columns], reverse=True))[:15]
    vals = [x for x, y in val_pairs]
    names = [y for x, y in val_pairs]
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    fig, ax = plt.subplots(figsize=(4,4))
    bars = ax.barh(names, vals)
    #ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.invert_yaxis()
    for i in range(len(names)):
        bars[i].set_color(colors[i % len(colors)])
        text = str(names[i])

        #plt.annotate(text, xy=(0, i), ha='left', va='center', color='black', weight='bold')
        plt.annotate(text, xy=(0, i), ha='left', va='center', color='black')
    if name is not None:
        fig.savefig(f'{dir}/{name}.png', bbox_inches='tight')
    if show:
        plt.show()


def compare_shaps_ban_l1_l2(feats_importances):
    options_ban = ['shap_simple', 'shap_fast', 'shap_brute', 'shap_orig_c', 'shap_orig', 'banzhaf_simple', 'banzhaf_fast']
    options_ban = set(options_ban) & set(feats_importances.keys())

    def l1(option_a, option_b):
        a = feats_importances[option_a]
        b = feats_importances[option_b]
        return np.sum(np.nan_to_num(np.abs((a - b).values)), nan=0, posinf=0, neginf=0)

    def l2(option_a, option_b):
        a = feats_importances[option_a]
        b = feats_importances[option_b]
        x = np.nan_to_num(np.abs((a - b).values))
        return np.sqrt(np.sum(x *x))


    logging.info('\n\nl1\n\t'+str(options_ban))
    for o in options_ban:
        logging.info(o+ str([f'{l1(o, o_b):0.2f}' for o_b in options_ban]))
    logging.info('\n\nl2\n\t'+str(options_ban))
    for o in options_ban:
        logging.info(o+ str([f'{l2(o, o_b):0.2f}' for o_b in options_ban]))

def compare_shaps(feats_importances):
    logging.info("************* comparing values using MAPE")
    options= ['shap_simple', 'shap_fast', 'shap_brute', 'shap_orig_c', 'shap_orig', 'banzhaf_simple', 'banzhaf_fast']
    options= set(options) & set(feats_importances.keys())

    def percantage_error(option_a, option_b):
        a = feats_importances[option_a]
        b = feats_importances[option_b]
        return np.nan_to_num(np.abs((a - b) / b).values, nan=0, posinf=0, neginf=0)


    logging.info('max\n\t'+str(options))
    for o in options:
        logging.info(o+ str([f'{np.max(percantage_error(o, o_b)):0.2f}' for o_b in options]))
    logging.info('\n\nargmax\n\t'+str(options))
    for o in options:
        logging.info(o+ str([f'{np.argmax(percantage_error(o, o_b)):0.2f}' for o_b in options]))
    logging.info('\n\nmean\n\t'+str(options))
    for o in options:
        logging.info(o+ str([f'{np.mean(percantage_error(o, o_b)):0.2f}' for o_b in options]))



def minSwaps(lst1, lst2):
    lst3 = []

    for i in range(len(lst1)):
        if lst1[i] in lst2:
            lst3.append(lst2.index(lst1[i]))

    return calculateSwapsToSort(lst3 + [i for i,e in enumerate(lst2) if e not in lst1])


def calculateSwapsToSort(lst):
    numOfSwaps = 0
    for index in range(len(lst)):
        if index != lst[index]:
            whereIsIndexMatchingNum = lst.index(index)
            lst[index], lst[whereIsIndexMatchingNum] = lst[whereIsIndexMatchingNum], lst[index]

            numOfSwaps +=1
        pass
    return numOfSwaps

def get_list(series):
    return series.abs().sort_values(ascending=False).keys().values


def compare_transpositions(feats_importances, NAME, limit=20, plot=False):
    logging.info(f"************* comparing values using transpositions limit = {limit}")
    filename=f'transpositions_cnt_{NAME}_{limit}'
    options = list(feats_importances.keys())
    n = len(feats_importances[options[0]])
    seed(0)
    idx = sample(range(n), min(1000, n))

    def transpositions(option_a, option_b):
        res = []
        for i in idx:
            l1 = get_list(feats_importances[option_a].iloc[i])[:limit]
            l2 = get_list(feats_importances[option_b].iloc[i])[:limit]
            res.append(minSwaps(l1.tolist(), l2.tolist()))
        return res

    transpositions_data = {}

    for o_a in tqdm(options):
        for o_b in options:
            v = transpositions(o_a, o_b)
            transpositions_data[(o_a,o_b)] = v
            bad_id = idx[np.argmax(v)]
            if plot:
                for o in [o_a, o_b]:
                    plot_vals(feats_importances[o].loc[bad_id:bad_id], f'{o}_{limit}', dir=f'bad/{NAME}/[{o_a}_vs_{o_b}]/')

    for fname,f in [('mean', np.mean), ('max', np.max)]:
        logging.info('fname\n\t'+str(options))
        for o in options:
            logging.info(o+ str([f'{f(transpositions_data[(o, o_b)]):0.2f}' for o_b in options]))

    logging.info('argmax\n\t'+str(options))
    for o in options:
        logging.info(o+ str([idx[np.argmax(transpositions_data[(o, o_b)])] for o_b in options]))

    os.makedirs('transpositions', exist_ok=True)
    with open(f'transpositions/{filename}.pickle', 'wb') as handle:
        pickle.dump(transpositions_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'transpositions/{filename}_idx.pickle', 'wb') as handle:
        pickle.dump(idx, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return transpositions_data,idx




