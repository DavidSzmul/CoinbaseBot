from Algorithms.portfolio import Portfolio
import config

import os
import pandas as pd
import numpy as np
import json
from matplotlib import pyplot as plt
from sklearn.preprocessing import Normalizer
from scipy import signal
# Algorithm made for 60s of resolution

class Simple_Algo(object):

    def __init__(self,
                    duration_historic=60, prc_taxes=0.01):
                 
        ## Properties 
        self.duration_historic = duration_historic # Default 1 hour
        self.prc_taxes = prc_taxes

        ## Variables
        self.reset()
    
    def reset(self):
        self.save = None
        self.transaction = None
        self.last_quality_factor = None

    def train(self):
        pass

    def test(self, portfolio, df_historic, verbose=0):
        
        ### Initialization
        # Generate database needed for running
        if len(df_historic)<self.duration_historic:
            raise ValueError('Historic size is too low to test the algorithm')
        if verbose: # Init displaying
            portfolio_historic = {}
        
        ### Loop
        for i in range(self.duration_historic, len(df_historic)):
            df_cut = df_historic.iloc[i-self.duration_historic:i, :]
            self.run(portfolio, df_cut, time=df_historic.index[i])
            if verbose:
                self.append_portfolio_historic(portfolio_historic, df_historic.index[i], portfolio, self.transaction)

        ### Display results
        if verbose:
            self.display(portfolio_historic)
            self.save = portfolio_historic
        pass

    def loop_RealTime(self, portfolio, verbose=0):

        # TODO Enabling the possibility to implement on local server the visualization
        # Create LocalSever

        while True:
            df_cut = df_historic.iloc[i-self.duration_historic:i, :]

    def run(self, portfolio, historic, time=None, debug_mode=False):
        
        #Initialization
        self.transaction = None
        matrix_in = historic.to_numpy()
        matrix_denoised = np.zeros(np.shape(matrix_in))
        matrix_denoised_normalized  = np.zeros(np.shape(matrix_in))
        matrix_percentage = np.zeros(np.shape(matrix_in))
        quality_factor = np.zeros((np.shape(matrix_in)[1],))

        keys_historic = list(historic.keys())
        keys_portfolio = list(portfolio.keys())

        ### Update prices of portfolio based on last time in historic
        portfolio.update_last_prices(historic.iloc[-1, :])

        ### Parameters
        Fs = 1/60 #Corresponding to 1min
        Fc = Fs/20
        Order = 2
        b,a = signal.butter(Order, Fc, 'low', fs=Fs, output='ba')
        pole = np.exp(-1/20)

        ### Determine the current crypto where all money is put      
        idx_from = np.argmax([portfolio[c]['value'] for c in keys_portfolio])
        from_ = keys_portfolio[idx_from]
        idx_historic_from = keys_historic.index(from_)

        ### PROCESSING
        for i in range(np.shape(matrix_in)[1]):
            ### Denoise to remove high variation
            matrix_denoised[:,i] = signal.filtfilt(b,a, matrix_in[:,i])

            ### Normalization in order to stay equivalent from one crypto to another
            matrix_denoised_normalized[:,i] = Normalizer().fit_transform([matrix_denoised[:,i]])[0]
        if np.any(matrix_denoised_normalized==0):
            raise ValueError('Null value that shouldn''t')

        ### Evolution of cryptos based on the current chosen one
        ### What is the most interesting is the derivative of evolution compared to the current crypto used
        ### Need also a middle term derivative in order verify if a local decrease is only temporary 
        NB_DELAY = 10 # In min
        diff_from = deriv_filter(matrix_denoised_normalized[:,idx_historic_from], NB_DELAY)
        for i in range(np.shape(matrix_in)[1]):
            # matrix_percentage[:,i] = matrix_denoised[:,i]/np.min(matrix_denoised[:,i]) -1 
            # matrix_percentage[:,i] = matrix_denoised[:,i]/np.average(matrix_denoised[:,i]) -1 
            # matrix_percentage[:,i] = matrix_denoised_normalized[:,i]/(matrix_denoised_normalized[:,idx_historic_from]) -1 
            matrix_percentage[:,i] = deriv_filter(matrix_denoised_normalized[:,i], NB_DELAY)

            ### Proposition of quality factor based on percentage
            in_factor = matrix_percentage[-np.shape(matrix_in)[0]//2:,i]
            ratio = np.ones(np.shape(in_factor))
            for j in (range(len(in_factor)-1)[::-1]):
                ratio[j] = pole*ratio[j+1]
            quality_factor[i] = sum(ratio*in_factor)/sum(ratio)*10000

        # Save quality for display
        self.last_quality_factor = {}
        ctr = 0
        for k in keys_historic:
            self.last_quality_factor[k] = quality_factor[ctr]
            ctr+=1
        
        #############################################################
        #### For the moment to simplify, only one crypto is chosen
        #############################################################

        value = portfolio[from_]['value']
        idx_to = np.argmax(quality_factor) # Index where to put all money
        to_ = keys_historic[idx_to]
        quality_to = quality_factor[idx_to]
        quality_from = quality_factor[idx_historic_from]

        stable_coin = 'USDC-USD'
        THR_ALLOW_CONVERSION = 40
        THR_STABLE = 10

        if (quality_to - quality_from>THR_ALLOW_CONVERSION) and (from_ != to_): #Allow transaction
            portfolio.convert_money(from_, to_, value, prc_taxes=self.prc_taxes)
            self.transaction = {'from': from_, 'to': to_, 'value': value, 'time': time}    
            print(self.transaction)

        elif (from_ !=stable_coin and quality_to-quality_factor[keys_historic.index(stable_coin)] < THR_STABLE): 
            portfolio.convert_money(from_, stable_coin, value, prc_taxes=self.prc_taxes)
            self.transaction = {'from': from_, 'to': stable_coin, 'value': value, 'time': time}    
            print(self.transaction)

        if debug_mode:
            t=np.linspace(0, np.shape(matrix_in)[0]-1, np.shape(matrix_in)[0])

            NB_SP = 4
            fig, sp = plt.subplots(NB_SP,1)
            for i in range(1,NB_SP-1):
                sp[0].get_shared_x_axes().join(sp[0], sp[i])

            matric_in_normalized = matrix_in.copy()
            for col in range(np.shape(matric_in_normalized)[1]):
                matric_in_normalized[:,col] = Normalizer().fit_transform([matric_in_normalized[:,col]])[0]
            
            sp[0].plot(t, matric_in_normalized, label=keys_historic)
            sp[1].plot(t, matrix_denoised_normalized, label=keys_historic)
            sp[2].plot(t, matrix_percentage, label=keys_historic)

            y_pos = np.arange(len(keys_historic))
            hbars = sp[3].barh(y_pos, quality_factor, align='center')
            sp[3].set_yticks(y_pos)
            sp[3].set_yticklabels(keys_historic)
            sp[3].invert_yaxis()  # labels read top-to-bottom
            sp[3].bar_label(hbars, fmt='%.2f')            
            sp[3].set_xlim(left=-1, right=1)  # adjust xlim to fit labels

            sp[0].set_title('Normalized Historic')
            sp[1].set_title('Normalized denoised Historic')
            sp[2].set_title('Percentage Evolution %')
            sp[2].set_xlabel('Time (min)')  
            sp[3].set_title('Quality factor')
            
            for i in range(3):
                sp[i].legend(loc="upper right")
            plt.show()  



    def append_portfolio_historic(self, historic, time, portfolio, transaction):

        #Initialization of dict
        if len(historic)==0:
            for k in portfolio.keys():
                historic[k] = {}
                for kk in portfolio[k].keys():
                    historic[k][kk] = []
            historic['time'] = []
            historic['transaction'] = []
            historic['quality_factor'] = pd.DataFrame()

        for k in portfolio.keys():
                for kk in portfolio[k].keys():
                    historic[k][kk].append(portfolio[k][kk])

        historic['time'].append(time)
        historic['transaction'].append((transaction is not None))
        historic['quality_factor']=historic['quality_factor'].append(self.last_quality_factor, ignore_index=True)

    def display(self, historic):
        NB_SP = 4
        fig, sp = plt.subplots(NB_SP, 1, sharex=True)

        ### Do copy of historic to remove useless cases
        hist = historic.copy()
        t = np.array(hist.pop('time'))
        t = np.floor(t/60)
        # t = (t-t[0])/60

        transaction_done = hist.pop('transaction')
        quality_factor = hist.pop('quality_factor')
        total_value = np.zeros(np.shape(t))

        for k in hist.keys():
            # # Verify that displaying crypto is relevant
            # if np.all(np.array(hist[k]['ammount'])==0):
            #     continue
            
            priceEvolution = np.array(hist[k]['last-price'])
            #########################
            ### Normalized Evolution
            priceEvolution_norm = Normalizer().fit_transform([priceEvolution])[0]
            sp[0].plot(t, priceEvolution_norm, '-*', label=k)
            #########################

            ### Values
            total_value += np.array(hist[k]['value'])
            sp[1].plot(t, hist[k]['value'], '-*', label=k)

            ### Quality factors
            sp[2].plot(t, quality_factor[k], '-*', label=k)
        ### Add total value
        sp[1].plot(t, total_value, '-*', label='Total')

        ### Confirmation of transaction done by algo
        sp[3].step(t, transaction_done, '-*', label='Transaction done')

        sp[0].set_ylabel('Normalized Price')
        sp[1].set_ylabel('Value Portfolio')
        sp[2].set_ylabel('Quality Factor')
        sp[-1].set_xlabel('Time (min)')

        for i in range(NB_SP):
            sp[i].legend()
        plt.tight_layout()
        plt.show()

def deriv_filter(x, nb_delay):
    b = [0]*(nb_delay+1)
    b[0] = 1
    b[-1] = -1
    a = [1]

    x_with_delay = np.concatenate((x[0]*np.ones((nb_delay,)), x), axis=0)
    diff_x = signal.lfilter(b,a, x_with_delay)
    return diff_x[nb_delay:]

if __name__ == '__main__':

    CRYPTO_STUDY_FILE = os.path.join(config.DATA_DIR, 'dtb/CRYPTO_STUDIED.json')
    STORE = os.path.join(config.DATA_DIR, 'dtb/store.h5')
    store = pd.HDFStore(STORE)
    df = store['min']

    # Take only crypto included inside Crypto_file
    with open(CRYPTO_STUDY_FILE) as f:
        data = json.load(f)
        crypto_study = [d['coinbase_name'] for d in data]
    crypto_to_remove = [c for c in df.columns if c not in crypto_study]
    df = df.drop(columns=crypto_to_remove)
    
    # #Cut
    # print('Cut done on database')
    # df = df.head(3500)

    Ptf = Portfolio()
    Ptf['USDC-USD']['last-price'] = 1
    Ptf.add_money(50, need_confirmation=False)
    Algo = Simple_Algo()

    # Algo.run(Ptf, df)
    Algo.test(Ptf, df, verbose=True)
    a=1
    
