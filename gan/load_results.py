"""
Michael S. Emanuel
Fri Dec  7 20:19:15 2018
"""

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_results(exp_name):
    """Load the results of one experiment from collection of json files output by bgan program."""
    # Path name for this experiment
    path = exp_name + "\\"
    # List of all results files
    fnames = os.listdir(path)
    # regex pattern for result files with names like "results_1600.json"
    pat = re.compile('results_(\d+)\.json')
    
    # Tables for the discriminant and generative losses
    disc_loss_trn = dict()
    disc_loss_tst = dict()
    gen_loss_trn = dict()
    gen_loss_tst = dict()
    
    # Iterate over result files
    for fname in fnames:
        regex_match = pat.match(fname)
        if regex_match is not None:
            iter_num = int(regex_match.group(1))
            with open(path + fname) as fh:
                data = json.load(fh)
            disc_losses = data['disc_losses']
            gen_losses = data['gen_losses']
            
            disc_loss_trn[iter_num] = disc_losses[0]
            disc_loss_tst[iter_num] = disc_losses[1]        
            gen_loss_trn[iter_num] = gen_losses[0]
            gen_loss_tst[iter_num] = gen_losses[1]
    
    # Arrays
    iter_num = np.array(sorted(disc_loss_trn.keys()))
    # Dataframe
    df = pd.DataFrame(data=iter_num, columns=['iter_num'])
    df['disc_loss_trn'] = np.array([disc_loss_trn[i] for i in iter_num])
    df['disc_loss_tst'] = np.array([disc_loss_tst[i] for i in iter_num])
    df['gen_loss_trn'] = np.array([gen_loss_trn[i] for i in iter_num])
    df['gen_loss_tst'] = np.array([gen_loss_tst[i] for i in iter_num])
    return df


def plot_results(df, exp_name):
    """Plot the results of one experiment stored in a dataframe."""
    fig, ax = plt.subplots(figsize=[12,8])
    ax.set_title(f'GAN Losses: {exp_name}')
    ax.set_xlabel('Iteration Number')
    ax.set_ylabel('Loss (%)')
    ax.plot(df.iter_num, df.disc_loss_trn)
    ax.plot(df.iter_num, df.disc_loss_tst)
    ax.plot(df.iter_num, df.gen_loss_trn)
    ax.plot(df.iter_num, df.gen_loss_tst)
    ax.legend()
    ax.grid()
    fname = f'{exp_name}.png'
    fig.savefig(fname)


def process(exp_name: str):
    """Process one experiment: load or generate the dataframe and plot it."""
    # Name of dataframe
    df_name = f'{exp_name}.csv'
    # Dataframe with results
    try:
        df = pd.read_csv(df_name)
    except:
        df = load_results(exp_name)
        df.to_csv(df_name)
        
    # Plot these results
    plot_results(df, f'{exp_name}')
    plot_results(df[0:100], f'{exp_name}_10k')


def main():
    # list of exeriments
    exp_names = ['four_shapes', 'mnist_semi_bayes', 'mnist_semi_ml']
    # process each experiment
    for exp_name in exp_names:
        process(exp_name)
        
if __name__ == '__main__':
    main()

