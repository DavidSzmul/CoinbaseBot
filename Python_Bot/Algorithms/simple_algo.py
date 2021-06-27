from Algorithms.portfolio import Portfolio
import config

import os
import pandas as pd
import numpy as np
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
            self.run(portfolio, df_cut)
            if verbose:
                self.append_portfolio_historic(portfolio_historic, df_historic.index[i], portfolio, self.transaction)
                self.transaction = None

        ### Display results
        if verbose:
            self.display(portfolio_historic)
            self.save = portfolio_historic
        pass

    def run(self, portfolio, historic,debug_mode=False):

        ### Update prices of portfolio based on last time in historic
        portfolio.update_last_prices(historic.iloc[-1, :])

        matrix_in = historic.to_numpy()
        matrix_denoised = np.zeros(np.shape(matrix_in))
        matrix_percentage = np.zeros(np.shape(matrix_in))
        quality_factor = np.zeros((np.shape(matrix_in)[1],))

        ### Filtering
        Fs = 1/60 #Corresponding to 1min
        Fc = Fs/8
        Order = 2
        b,a = signal.butter(Order, Fc, 'low', fs=Fs, output='ba')
        pole = np.exp(-1/20)

        keys_historic = list(historic.keys())
        keys_portfolio = list(portfolio.keys())

        for i in range(np.shape(matrix_in)[1]):

            ### Filtering to get percentage
            matrix_denoised[:,i] = signal.filtfilt(b,a, matrix_in[:,i])
            matrix_percentage[:,i] = matrix_denoised[:,i]/np.average(matrix_denoised[:,i]) -1 

            ### Proposition of quality factor based on percentage
            in_factor = matrix_percentage[-np.shape(matrix_in)[0]//2:,i]
            ratio = np.ones(np.shape(in_factor))
            for j in (range(len(in_factor)-1)[::-1]):
                ratio[j] = pole*ratio[j+1]
            quality_factor[i] = sum(ratio*in_factor)/sum(ratio)*100
        
        #############################################################
        #### For the moment to simplify, only one crypto is chosen
        #############################################################

        ### Determine the current crypto where all money is put
        idx_from = np.nonzero([portfolio[c]['value'] for c in keys_portfolio])[0][0]
        from_ = keys_portfolio[idx_from]
        value = portfolio[from_]['value']

        quality_from = quality_factor[keys_portfolio.index(from_)]

        idx_to = np.argmax(quality_factor) # Index where to put all money
        to_ = keys_historic[idx_to]
        quality_to = quality_factor[idx_to]

        THR_ALLOW_CONVERSION = 0.1
        if quality_to - quality_from>THR_ALLOW_CONVERSION and from_ != to_: #Allow transaction

            portfolio.convert_money(from_, to_, value, prc_taxes=self.prc_taxes)
            self.transaction = {'from': from_, 'to': to_, 'value': value}     
            print(self.transaction)

            if debug_mode:
                t=np.linspace(0, np.shape(matrix_in)[0]-1, np.shape(matrix_in)[0])

                fig, sp = plt.subplots(3,1,sharex=True)
                sp[0].plot(t, matrix_in, label='raw')
                sp[1].plot(t, matrix_denoised, label='hfilter')
                sp[2].plot(t, matrix_percentage, label='percentage')

                sp[-1].set_xlabel('Time (min)')  
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

        for k in portfolio.keys():
                for kk in portfolio[k].keys():
                    historic[k][kk].append(portfolio[k][kk])

        historic['time'].append(time)
        historic['transaction'].append((transaction is not None))

    def display(self, historic):
        NB_SP = 3
        fig, sp = plt.subplots(NB_SP, 1, sharex=True)

        ### Do copy of historic to remove useless cases
        hist = historic.copy()
        t = hist.pop('time')
        t = (t-t[0])/60
        transaction_done = hist.pop('transaction')
        total_value = np.zeros(np.shape(t))

        for k in hist.keys():
            # Verify that displaying crypto is relevant
            if np.all(np.array(hist[k]['ammount'])==0):
                continue
            
            
            priceEvolution = np.array(hist[k]['last-price'])
            #########################
            ### 1: Percentage price
            # pricePrc = np.diff(priceEvolution, prepend=priceEvolution[0])/priceEvolution
            # sp[0].plot(t, pricePrc, '-*', label=k)

            ### 2: Normalized Evolution
            priceEvolution_norm = Normalizer().fit_transform([priceEvolution])[0]
            sp[0].plot(t, priceEvolution_norm, '-*', label=k)
            #########################

            ### Values
            total_value += np.array(hist[k]['value'])
            sp[1].plot(t, hist[k]['value'], '-*', label=k)

        ### Confirmation of transaction done by algo
        sp[2].step(t, transaction_done, '-*', label='Transaction done')

        ### Add total value
        sp[1].plot(t, total_value, '-*', label='Total')

        sp[0].set_ylabel('Percentage Evolution (%)')
        sp[1].set_ylabel('Value Portfolio')
        sp[-1].set_xlabel('Time (min)')

        for i in range(NB_SP):
            sp[i].legend()

        plt.show()

if __name__ == '__main__':

    CRYPTO_STUDY_FILE = os.path.join(config.DATA_DIR, 'dtb/CRYPTO_STUDIED.json')
    STORE = os.path.join(config.DATA_DIR, 'dtb/store.h5')
    store = pd.HDFStore(STORE)
    df = store['min']

    Ptf = Portfolio()
    Ptf['USDC-EUR']['last-price'] = 0.84
    Ptf.add_money(50, need_confirmation=False)
    Algo = Simple_Algo()

    # Algo.run(Ptf, df)
    Algo.test(Ptf, df, verbose=True)
    a=1
    
