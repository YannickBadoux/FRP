import warnings
warnings.filterwarnings("ignore", module="matplotlib")

from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import umap
from astropy.io import fits
import time
import argparse
from sklearn.model_selection import train_test_split
from astropy.table import Table
from npy_append_array import NpyAppendArray
import glob

def distance(df)->np.ndarray:
    '''Calculate distance in parsecs from parallax in mas

    Parameters:
    df: DataFrame or Table containing the parallax values, with key: 'parallax
    '''
    try:
        d = 1/(1e-3*np.abs(df['parallax']))
    except KeyError:
        d = 1/(1e-3*np.abs(df['parallax_x']))
    return d

def abs_gmag(df)->np.ndarray:
    '''Calculate absolute G band magnitude
    
    Parameters:
    df: DataFrame or Table containing the apparent G band magnitude and parallax values, with key: 'phot_g_mean_mag', 'parallax'
    '''
    d = distance(df)
    try:
        absG = df["phot_g_mean_mag"] - 5*np.log10(d) + 5
    except KeyError:
        absG = df["phot_g_mean_mag_x"] - 5*np.log10(d) + 5
    return absG

if __name__ == '__main__':
    #parse arguments
    parser = argparse.ArgumentParser(description='UMAP projection of xp spectra')
    parser.add_argument('--nn', type=int, default=15, help='Number of nearest neighbours')
    parser.add_argument('--md', type=float, default=0, help='Minimum distance')
    parser.add_argument('--rs', type=int, default=2208, help='Random state')
    parser.add_argument('-run_number', type=int, default=0, help='Run number')
    parser.add_argument('-n_nodes', type=int, default=1, help='Number of nodes to run on in parallel')

    args = parser.parse_args()

    #path to the training data folder
    tr_path = 'training_data/'

    print('--------Importing and processing training data--------')
    start_time = time.time()
    #import and process random star sample
    rs_coeffs = pd.read_pickle(tr_path+"more_complete_sample_xp.pkl")
    rs_params = fits.open(tr_path+"more_complete_sample-result.fits")[1].data
    rs_params = pd.DataFrame(rs_params.newbyteorder().byteswap())
    labels = pd.read_csv(tr_path+"more_complete_sample.csv")
    labels.drop(columns = ['phot_g_mean_mag', 'bp_rp', 'parallax'], inplace=True)

    #merge both dfs and drop duplicates
    rs_coeffs = rs_coeffs.merge(rs_params, on='source_id')
    rs_coeffs = rs_coeffs.merge(labels, on='source_id')
    rs_coeffs.drop_duplicates(subset='source_id', inplace=True, ignore_index=True)

    rs_coeffs['label'] = 'other'
    rs_coeffs['label_num'] = 0
    rs_coeffs['src_df'] = 'rs'
    rs_coeffs[rs_coeffs['blue_giant'] == True]['label_num'] = 1

    #remove one star with missing bp coefficients
    rs_coeffs.dropna(subset=['bp_coefficients'], inplace=True, ignore_index=True)


    #import and process hot star sample
    hs_coeffs = Table.read(tr_path+"hot_stars_xp_coefficients.fits", format='fits').to_pandas()
    hs_params = fits.open(tr_path+"hot_stars_full-result.fits")[1].data
    hs_params = pd.DataFrame(hs_params.newbyteorder().byteswap())

    #merge both dfs and drop duplicates
    hs_coeffs = hs_coeffs.merge(hs_params, on='source_id')
    hs_coeffs.drop_duplicates(subset='source_id', inplace=True, ignore_index=True)

    #add labels
    hs_coeffs['label'] = 'hot_star'
    hs_coeffs['label_num'] = 1
    hs_coeffs['src_df'] = 'hs'


    #import and process hot and luminous main sequence stars
    hlms_coeffs = pd.read_pickle(tr_path+"hot_lum_ms_10k_xp.pkl")
    hlms_params = fits.open(tr_path+"hot_lum_ms_10k-result.fits")[1].data
    hlms_params = pd.DataFrame(hlms_params.newbyteorder().byteswap())

    #merge both dfs and drop duplicates
    hlms_coeffs = hlms_coeffs.merge(hlms_params, on='source_id')
    hlms_coeffs.drop_duplicates(subset='source_id', inplace=True, ignore_index=True)

    #add labels
    hlms_coeffs['label'] = 'other'
    hlms_coeffs['label_num'] = 1
    hlms_coeffs['src_df'] = 'hlms'

    #import and process BHB stars
    bhb_coeffs = Table.read(tr_path+'bhb_culpan23_coefficients.fits', format='fits').to_pandas()

    bhb_params = Table.read(tr_path+'BHB_culpan2023-result.fits', format='fits').to_pandas()

    #merge both dfs and drop duplicates
    bhb_coeffs = bhb_coeffs.merge(bhb_params, on='source_id')
    bhb_coeffs.drop_duplicates(subset='source_id', inplace=True, ignore_index=True)

    #add labels
    bhb_coeffs['label'] = 'BHB'
    bhb_coeffs['label_num'] = 0
    bhb_coeffs['src_df'] = 'bhb'

    #import and process random zari stars
    rz_coeffs = Table.read(tr_path+'random_zari_sample_coefficients.fits', format='fits').to_pandas()

    rz_params = Table.read(tr_path+'random_zari_sample-result.ecsv', format='ascii.ecsv').to_pandas()

    #merge both dfs and drop duplicates
    rz_coeffs = rz_coeffs.merge(rz_params, on='source_id')
    rz_coeffs.drop_duplicates(subset='source_id', inplace=True, ignore_index=True)

    #add labels
    rz_coeffs['label'] = 'rzs'
    rz_coeffs['label_num'] = 1
    rz_coeffs['src_df'] = 'rz'

    print('--------Importing done--------')


    #concatenate all dfs
    df = pd.concat([hs_coeffs, rz_coeffs, hlms_coeffs, rs_coeffs, bhb_coeffs], ignore_index=True)
    df['abs_gmag'] = abs_gmag(df)
    print('full sample shape:',df.shape)

    # normalise coefficients by dividing by the mean g band flux
    norm = 'phot_g_mean_flux' #key normalisation values

    #extract coefficients and get them in the correct shape
    bp_coeffs = df['bp_coefficients'].to_numpy()
    bp_coeffs = np.expand_dims(bp_coeffs, axis=1)
    bp_coeffs = np.vstack([bp[0].data for bp in bp_coeffs])

    rp_coeffs = df['rp_coefficients'].to_numpy()
    rp_coeffs = np.expand_dims(rp_coeffs, axis=1)
    rp_coeffs = np.vstack([rp[0].data for rp in rp_coeffs])

    #divide by the mean g band flux
    bp_coeffs_norm = bp_coeffs / np.expand_dims(df[norm].to_numpy(), axis=1)
    rp_coeffs_norm = rp_coeffs / np.expand_dims(df[norm].to_numpy(), axis=1)

    #add to df
    df['bp_coefficients_norm'] = bp_coeffs_norm.tolist()
    df['rp_coefficients_norm'] = rp_coeffs_norm.tolist()

    #construct 2d array of normalised coefficients
    norm_coeffs = np.concatenate((bp_coeffs_norm,rp_coeffs_norm), axis=1)

    #split the df into test and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df, df['label_num'], test_size=0.2, random_state=args.rs)
    train_idx, test_idx = X_train.index, X_test.index

    norm_coeffs_train = norm_coeffs[train_idx]
    norm_coeffs_test = norm_coeffs[test_idx]
    print('Train sources:',norm_coeffs_train.shape,'Test sources:', norm_coeffs_test.shape)

    end_time = time.time()
    print('importing and processing training data time:', end_time-start_time)


    #----------------------------------UMAP projection----------------------------------
    print(f'Fitting UMAP model with: nn = {args.nn}, md = {args.md}, rs = {args.rs}')

    data = norm_coeffs_train #training data
    test_data = norm_coeffs_test #testing data

    mtitle = 'normalized coefficients'

    reducer = umap.UMAP(n_neighbors = args.nn, min_dist = args.md, random_state=args.rs)

    #fit model
    start_time = time.time()
    reducer.fit(data)
    print('Ftting done')
    print('Time taken:', time.time()-start_time)

    #project training data using the fitted model
    data_to_be_projected = data #data to be projected (the model wont be trained on this data)
    embedding = reducer.transform(data_to_be_projected)
    test_embedding = reducer.transform(test_data)

    #if it is the first run, save the train and test embedding and plot them
    if args.run_number == 1:
        np.save(f'UMAP_embeddings/train_embedding_nn{args.nn}_md{args.md}_rs{args.rs}.npy', embedding)
        np.save(f'UMAP_embeddings/test_embedding_nn{args.nn}_md{args.md}_rs{args.rs}.npy', test_embedding)
        print('Projection done')

        #plot training data
        tr_hot_mask = (X_train['src_df'] == 'rz') | (X_train['src_df'] == 'hs') | (X_train['src_df'] == 'hlms') | (X_train['blue_giant'])
        tr_bhb_mask = (X_train['src_df'] == 'bhb')
        tr_rs_mask = (X_train['src_df'] == 'rs') & (X_train['blue_giant']==False)

        #plot the training data
        fig, axs = plt.subplots(2,2,figsize=(20,22), height_ratios=[1,1.1], sharex=True, sharey=True)
        fig.suptitle(f'UMAP projection of training {mtitle} with nn = {args.nn}, md = {args.md}')

        #labels
        axs[0,0].scatter(embedding[tr_hot_mask,0], embedding[tr_hot_mask,1], s=3, zorder=100, label = 'Hot stars')
        axs[0,0].scatter(embedding[tr_bhb_mask,0], embedding[tr_bhb_mask,1], s=3, zorder=99, label = 'BHB stars')
        axs[0,0].scatter(embedding[tr_rs_mask,0], embedding[tr_rs_mask,1], s=1, zorder=1, label = 'Other', alpha=0.3)
        axs[0,0].legend(loc='best')
        axs[0,0].set_title('Labels')

        #density
        axs[0,1].hexbin(embedding[:,0], embedding[:,1], gridsize=70, cmap='gist_heat_r')
        axs[0,1].set_title('Density')

        #absolute G magnitude
        axs[1,0].scatter(embedding[:,0], embedding[:,1], s=1, c=abs_gmag(X_train), cmap='plasma_r', vmin=-3, vmax=15)
        axs[1,0].set_title('Absolute G magnitude')
        fig.colorbar(axs[1,0].collections[0], ax=axs[1,0], label='Absolute G magnitude', location='bottom', fraction=0.046, pad=0.04)

        #bp-rp
        axs[1,1].scatter(embedding[:,0], embedding[:,1], s=1, c=X_train['bp_rp'], cmap='coolwarm', vmin=0, vmax=5)
        axs[1,1].set_title('BP-RP')
        fig.colorbar(axs[1,1].collections[0], ax=axs[1,1], label='BP-RP', location='bottom', fraction=0.046, pad=0.04)

        fig.tight_layout()
        plt.savefig(f'UMAP_figures/all_sources_phot_g_nn{args.nn}_md{args.md}_rs{args.rs}_train.png', bbox_inches='tight')



        #plot test data
        test_hot_mask = (X_test['src_df'] == 'rz') | (X_test['src_df'] == 'hs') | (X_test['src_df'] == 'hlms') | (X_test['blue_giant'])
        test_bhb_mask = (X_test['src_df'] == 'bhb')
        test_rs_mask = (X_test['src_df'] == 'rs') & (X_test['blue_giant']==False)

        #plot the test data
        fig, axs = plt.subplots(2,2,figsize=(20,22), height_ratios=[1,1.1], sharex=True, sharey=True)
        fig.suptitle(f'UMAP projection of test {mtitle} with nn = {args.nn}, md = {args.md}')

        #labels
        axs[0,0].scatter(test_embedding[test_hot_mask,0], test_embedding[test_hot_mask,1], s=3, zorder=100, label = 'Hot stars')
        axs[0,0].scatter(test_embedding[test_bhb_mask,0], test_embedding[test_bhb_mask,1], s=3, zorder=99, label = 'BHB stars')
        axs[0,0].scatter(test_embedding[test_rs_mask,0], test_embedding[test_rs_mask,1], s=1, zorder=1, label = 'Other', alpha=0.3)
        axs[0,0].legend(loc='best')
        axs[0,0].set_title('Labels')

        #density
        axs[0,1].hexbin(test_embedding[:,0], test_embedding[:,1], gridsize=70, cmap='gist_heat_r')
        axs[0,1].set_title('Density')

        #absolute G magnitude
        axs[1,0].scatter(test_embedding[:,0], test_embedding[:,1], s=1, c=abs_gmag(X_test), cmap='plasma_r', vmin=-3, vmax=15)
        axs[1,0].set_title('Absolute G magnitude')
        fig.colorbar(axs[1,0].collections[0], ax=axs[1,0], label='Absolute G magnitude', location='bottom', fraction=0.046, pad=0.04)

        #bp-rp
        axs[1,1].scatter(test_embedding[:,0], test_embedding[:,1], s=1, c=X_test['bp_rp'], cmap='coolwarm', vmin=0, vmax=5)
        axs[1,1].set_title('BP-RP')
        fig.colorbar(axs[1,1].collections[0], ax=axs[1,1], label='BP-RP', location='bottom', fraction=0.046, pad=0.04)

        fig.tight_layout()
        plt.savefig(f'UMAP_figures/all_sources_phot_g_nn{args.nn}_md{args.md}_rs{args.rs}_test.png', bbox_inches='tight')

    #project all data using the fitted model
    idx = args.run_number-1

    #path to folder with normalised coefficients
    norm_coeff_path = 'norm_coefficients'

    flist = glob.glob(norm_coeff_path)[idx::args.n_nodes] #every nth file

    #save path
    save_path = ''

    for i,f in enumerate(flist):
        print(f'Processing {f}')
        data = np.load(f)
        embedding = reducer.transform(data[:,1:]) #project data into the learnt space

        #save the embedding
        tab = Table([data[:,0], embedding], names=['source_id', 'embedding'], dtype=[np.int64, np.ndarray])
        Table.write(tab, save_path + f'{idx}_{i}_embedding.fits', format='fits')