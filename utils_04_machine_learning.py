# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 23:23:13 2021

@author: saulg
"""
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pickle
import os
from scipy import interpolate
from sklearn.metrics import mean_squared_error
import gc

class imputation():
    def __init__(self, data_root ='./Datasets', figures_root = './Figures Imputed'):
        # Data Path
        if os.path.isdir(data_root) is False:
            os.makedirs(data_root)
            print('The dataset folder with data is empty')
        self.data_root = data_root
        
        # Fquifer Root is the location to save figures.
        if os.path.isdir(figures_root) is False:
            os.makedirs(figures_root)
        self.figures_root = figures_root
        return
       
    def read_pickle(self, pickle_file, pickle_root):
        wellfile = pickle_root + pickle_file + '.pickle'
        with open(wellfile, 'rb') as handle:
            wells = pickle.load(handle)
        return wells
    
    def Save_Pickle(self, Data, name:str, path:str, protocol:int = 3):
        with open(path + '/' + name + '.pickle', 'wb') as handle:
            pickle.dump(Data, handle, protocol=protocol)
            
    def log_errors(self, errors, name:str, path:str):
        if len(errors) == 0: pass
        else:
            with open(path + '/' + name + '.txt', 'w') as output:
                for row in errors:
                    output.write(str(row) + '\n')
    
    def Data_Split(self, Data, well_name_temp, Shuffle=False):
        if Shuffle:
            # The frac keyword argument specifies the fraction of rows
            # to return in the random sample
             Data = Data.sample(frac=1)
        Y = Data[well_name_temp].to_frame()
        X = Data.drop(well_name_temp, axis=1)
        return Y, X
    
    def Data_Join(self, pd1, pd2, method='outer', axis=1):
        return pd.concat([pd1, pd2], join = method, axis=1)
    
    def test_range_split(self, df, f_index, name, min_points = 1, 
                         cut_left= None, gap_year=None, random=True, 
                          seed_start = 42, max_tries = 15, max_gap = 5):
        self.gap_year = gap_year
        self.cut_left = cut_left
        attempt = 0
        Y_Test = []
        Y_Train = []
        while attempt <= max_tries and len(Y_Test) <= min_points and len(Y_Train) <= 3:
            if self.gap_year == None: 
                gap_year = np.random.randint(1, max_gap+1)
            if self.cut_left == None: 
                cut_left = None
            cut_left, cut_right = self.define_gap(df, f_index, cut_left, gap_year, 
                                                  seed = seed_start + attempt, 
                                                  random = random)
            cut_left_index = dt.datetime(cut_left, 1, 1)
            cut_right_index = dt.datetime(cut_right, 1, 1)
            Y_Test = df[(df.index >= cut_left_index) & 
                    (df.index < cut_right_index)].dropna() 
            Y_Train = df[(df.index < cut_left_index) |
                    (df.index >= cut_right_index)].dropna() 
            attempt += 1
        if attempt == max_tries:
            self.cut_left, self.cut_right= [f_index[0], f_index[0]]
            error = f'{name} could not be split into train/test sets'
            return df.dropna(), pd.DataFrame(), error
        self.cut_left, self.cut_right= [cut_left, cut_right]
        return Y_Train, Y_Test, None
    
    def define_gap(self, df, f_index, cut_left, gap_year, seed, random=True):
        np.random.seed(seed)
        df = df.dropna()
        if random == True and cut_left == None:
            if df.index[0].year < f_index[0].year: date_min = f_index[0].year
            else: date_min = df.index[0].year
            if df.index[-1].year > f_index[-1].year: cut_range_end = f_index[-1].year
            else: cut_range_end = df.index[-1].year
            
            df_index = df[(df.index >= dt.datetime(date_min, 1, 1)) &
                          (df.index <= dt.datetime(cut_range_end - gap_year, 1, 1))]
            cut_left = df.index[np.random.randint(0, len(df_index))].year
            cut_right = cut_left + gap_year
        elif random == False and cut_left != None:
            cut_right = cut_left + gap_year  
        else: cut_left, cut_right = [1995, 2001] 
        return cut_left, cut_right
    
    def interpolate(self, feature_index, y, name, shift = 60):
        self.shift_rw_max = shift
        index_check = y.dropna()
        startdt = pd.to_datetime(index_check.index[0])
        enddt   = pd.to_datetime(feature_index[0])
        lag = len(pd.date_range(start=startdt,end=enddt,freq='M'))
        if lag > 0: y = index_check
        int_y, x_int_index = self._interpolate_integers(lag, feature_index, y)
        
        p_class = interpolate.pchip(int_y.index, int_y[name], extrapolate=False)
        pchip_index = feature_index.shift(shift-1, freq='MS').union(feature_index.shift(-shift, freq='MS'))
        pchip_int   = np.arange(-shift, len(pchip_index)-shift, 1).astype(int)
        pchip_int   = pd.DataFrame(pchip_int, index = pchip_index, columns = ['x'])
        interp  = p_class(pchip_index[:].astype('int'))
        pchip   = pd.DataFrame(interp, index=pchip_index[:], columns=['pchip'])
        x_int_index = self.Data_Join(x_int_index, pchip)
        x_int_index = x_int_index.drop('pchip', axis=1)
        x_int_index = x_int_index['x'].fillna(pchip_int['x'])
        return pchip, x_int_index, pchip_int
    
    def _interpolate_integers(self, lag, feature_index, data):
        data = data.dropna()
        start = min(feature_index[0], data.index[0])
        end = max(feature_index[-1], data.index[-1])
        data_range = pd.date_range(start= start, end = end, freq='MS')
        int_values = np.arange(-lag, len(data_range)-lag, 1).astype(int)
        x_index    = pd.Series(int_values, index = data_range, name = 'x')
        int_y      = pd.concat([x_index, data], axis=1).dropna()
        return int_y, x_index
    
    def linear_extrap(self, f_index, y, shift, reg_perc = [1.0, 0.5, 0.25, 0.10], max_sd = 3 , outlier = 3, force_left = False, force_right=False, cSlope_left=False, cSlope_right=False):
        # Generate extrapolation df for left and right sides
        left = pd.DataFrame(index = f_index.index, columns = reg_perc)
        right = pd.DataFrame(index = f_index.index, columns = reg_perc)
        # Add df to dict to be iteratable 
        d_side = {"left":left, "right":right}
        # Dictionary to store metadata about slope corrections
        d_slope = {}
        # Find smallest slope percentage to save index for linear regression
        # and data to determine y-intercept
        s_min = min(reg_perc)
        
        # High-Level: Generate slope metadata df with S_raw, Int, Mean, Slope
        for i, side in enumerate(d_side):
            slope_df = pd.DataFrame(index = reg_perc, columns = ['Slope', 'Int', 'Mean'])
            for j, perc in enumerate(reg_perc):
                points = int(len(y.dropna()) * perc)
                
                # Obtain the percentage of points in each side
                # Grab all indecies associated with the data range
                # reindex data so that regression makes sense
                if side == "left":
                    data = y[:points]
                    index = f_index[f_index.index <= data.index[-1]]
                    index = pd.DataFrame(np.arange(points - len(index), points, 1), index = index.index, columns = ['x'])
                    if perc == s_min:
                        index_l = index
                
                elif side == "right":
                    data = y[-points:]
                    index = f_index[f_index.index >= data.index[0]]
                    index = pd.DataFrame(np.arange(len(index) - len(index), len(index), 1), index = index.index, columns = ['x'])
                    if perc == s_min:
                        index_r = index

                # Get sample Mean and Standard Deviation
                mean = data['pchip'].mean()
                sd = data['pchip'].std()
                # Remove outliers for linear regression
                data = data[(data['pchip'] <= mean + outlier * sd)]
                # Perform Linear Regression
                reg = self.linear_regression(index, data)
                # Unpack slope and intercept
                slope, intercept, _ = reg
                # Get Mean of data set once outliers are removed
                mean = data['pchip'].mean()
                # Save slope mean and intercept
                slope_df.loc[perc] = np.array([slope, intercept, mean])
                # Extrapolate based on slope store extrapolation
                d_side[side][perc] = index * slope + intercept
                
            # Update datafrme in dictionary
            d_slope[side] = slope_df
            # Get average slope for each side
            if side == 'left': slope_l = slope_df['Slope'].mean()
            elif side == 'right': slope_r = slope_df['Slope'].mean()
        
        # Calculate population Mean, Std, Max value, Min Value
        pop_mean = y['pchip'].mean()
        pop_std = y['pchip'].std()
        max_value = pop_mean + max_sd * pop_std
        min_value = pop_mean - max_sd * pop_std
        
        # Check if we're forcing prior different than data: Left
        if force_left == 'positive': slope_l = abs(slope_l) 
        elif force_left == 'negative': slope_l = -1 * abs(slope_l)
        elif cSlope_left != False: slope_l = cSlope_left
        
        # Check if we're forcing prior different than data: Right
        if force_right == 'positive': slope_r = abs(slope_r) 
        elif force_right == 'negative': slope_r = -1 * abs(slope_r)
        elif cSlope_right != False: slope_r = cSlope_right
        
        # Extrapolate based on mean slope
        extrap_l = index_l * slope_l + d_slope['left']['Mean'].loc[s_min]
        extrap_r = index_r * slope_r + d_slope['right']['Mean'].loc[s_min]
        
        # Replace extrapolated values that exceed limits
        # Correct Left Side
        extrap_l['x'] = np.where(extrap_l['x'] <= max_value, extrap_l['x'], max_value)
        extrap_l['x'] = np.where(extrap_l['x'] >= min_value, extrap_l['x'], min_value)
        
        # Correct Right Side
        extrap_r['x'] = np.where(extrap_r['x'] <= max_value, extrap_r['x'], max_value)
        extrap_r['x'] = np.where(extrap_r['x'] >= min_value, extrap_r['x'], min_value)
        
        # Fill Extrapolation
        extrap_df = self.Data_Join(f_index, y).drop(['x'], axis=1)
        filled = extrap_df['pchip'].fillna(extrap_l['x'])
        filled = filled.fillna(extrap_r['x'])
        return filled, d_side, d_slope

    def hampel_filter(self, df_imp, df_obs, max_sd  = 3, window = 36, center = True):
        # set dataframe filtering limits
        left = max(df_imp.index[0], df_obs.index[0])
        right = min(df_imp.index[-1], df_obs.index[-1])
        
        # Create empty df same size as imputation df
        df = pd.DataFrame(index = df_imp.index, columns = df_imp.columns)
        # Filter observed values within data range
        df_obs = df_obs.loc[(df_obs.index >= left) & (df_obs.index <= right), :]
        # Fill df to ensure hampel doesn't remove observed measurements
        df = df.fillna(df_obs)
        
        # Standard deviation approximation
        L = 1.4826
        
        # Copy data
        data = df_imp.copy()
        
        # Generate find average median Done because edges will not have data
        if center == True:
            roll = data.rolling(window=window, min_periods = 1, center=True).median()
        elif center == False:
            roll = data.rolling(window=window, min_periods = 1, center=False).median()
        
        # Applying filter
        difference = np.abs(roll - data)
        mad = difference.rolling(window, min_periods = 1).median()
        threshold = max_sd * L * mad
        outlier_idx = difference > threshold
        data[outlier_idx] = roll # np.nan #roll_median #
        
        # Fill observed data with filtered measurements
        df = df.fillna(data).astype(float)
        return df
    
    def smooth(self, df_imp, df_obs, window = 36, center = True):
        # set dataframe filtering limits
        left = max(df_imp.index[0], df_obs.index[0])
        right = min(df_imp.index[-1], df_obs.index[-1])
        
        # Create empty df same size as imputation df
        df = pd.DataFrame(index = df_imp.index, columns = df_imp.columns)
        # Filter observed values within data range
        df_obs = df_obs.loc[(df_obs.index >= left) & (df_obs.index <= right), :]
        # Fill df to ensure hampel doesn't remove observed measurements
        df = df.fillna(df_obs)
        # Copy data
        data = df_imp.copy()
        # Generate find average median Done because edges will not have data
        if center == True:
            roll = data.rolling(window=window, min_periods = 1, center=True).median()
        elif center == False:
            roll = data.rolling(window=window, min_periods = 1, center=False).median()
        # Fill observed data with filtered measurements
        df = df.fillna(roll).astype(float)
        return df
        
    
    def linear_regression(self, f_index, y):
        df = self.Data_Join(f_index, y).dropna()
        x = df['x'].values
        N = len(x)
        y = df['pchip'].values
        one = np.ones((N))
        if len(x) == len(y) and len(x) >= 3:
            A      = np.vstack([x,one]).T
            B0, B1 = np.linalg.lstsq(A, y)[0]
        reg_line = 'y = {}X + {}'.format(B0, B1)
        return (B0, B1, reg_line)
    
    def rolling_windows(self, df, windows = [3, 6]):
        rw_dict = dict()
        for i, months in enumerate(windows):
            key = str(months) + 'm_rw'
            rw = df.rolling(months, center=True, min_periods = 1).mean()
            rw_dict[key] = rw
        rw = pd.DataFrame.from_dict(rw_dict)
        return rw
        
    def scaler_pipline(self, x, scaler, ns_names, train=False):
        if train == True:
            x_scale = x.drop(ns_names, axis=1)
            x_temp = scaler.fit_transform(x_scale)
            x_temp = pd.DataFrame(x_temp, index = x_scale.index, columns = x_scale.columns)
            X = self.Data_Join(x_temp, x[ns_names], method='inner')
            return [X, scaler]
            
        else:
            x_scale = x.drop(ns_names, axis=1)
            x_temp = scaler.transform(x_scale)
            x_temp = pd.DataFrame(x_temp, index = x_scale.index, columns = x_scale.columns)
            X = self.Data_Join(x_temp, x[ns_names], method='inner')
            return X
    
    def feature_correlation(self, df, Feature_Data, raw, score):
        data_clean = Feature_Data.dropna()
        fi = len(Feature_Data)
        wi = len(data_clean)
        data_zc = data_clean - data_clean.mean()
        index = Feature_Data.columns[0]
        df.loc[index, 'FI'] = fi
        df.loc[index, 'WI'] = wi
        names = Feature_Data.columns[1:].tolist()
        score = score['w_score'].tolist()
        s_list = list(zip(names, score))
        s_list.sort(reverse=True, key = lambda x: x[1])
        for i, obj in enumerate(s_list):
            name, r = obj
            if raw.index[-1] > Feature_Data.index[-1]: 
                index_mask = Feature_Data.index
            else: 
                index_mask = pd.date_range(Feature_Data.index[0], raw.index[-1], freq='MS')
            index_mask = index_mask[index_mask.isin(raw[name].index)]
            fip = raw[name].loc[index_mask].count()/fi
            wip = raw[name].loc[Feature_Data.dropna().index].count()/wi
            rmse  = mean_squared_error(data_zc[index].values, data_zc[name].values, squared=False)
            output = [name, r, fip, wip, rmse]
            col_names = [f'F{i}', f'F{i} w_r2', f'F{i} fip', f'F{i} wip', f'F{i} nRMSE']
            if col_names[0] not in df.columns:
                df[col_names] = np.nan
            df.loc[index, col_names] = output
        return df
    
    def metrics(self, Metrics, n_wells):
        metrics_result = Metrics.sum(axis = 0)
        normalized_metrics = metrics_result/n_wells
        with open(self.figures_root + 'Error_Aquifer.txt', "w") as outfile:
            print('Raw Model Metrics:  \n' + str(metrics_result), file=outfile)
            print('\n Normalized Model Metrics Per Well:  \n' + str(normalized_metrics), file=outfile)
        np.savetxt(self.figures_root + 'Error_Wells.txt', Metrics.values, fmt='%d')
        return metrics_result, normalized_metrics

    def Model_Training_Metrics_plot(self, Data, name, show=False):
        fig = plt.figure()
        for key in Data: plt.plot(Data[key])
        plt.grid(True)
        plt.gca().set_ylim(-0.5, 5)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(Data.keys())
        plt.savefig(self.figures_root + '/' + name + '_Training_History')
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()
    
    def trend_plot(self, pchip, extrap_df, extrap_md, raw, name, extension = '.png', show=False):
        slopes_l = extrap_df['left']
        slopes_r = extrap_df['right']
        meta_l = extrap_md['left']
        meta_r = extrap_md['right']
        num_plots = len(slopes_l.columns)
        
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(111)
        
        colormap = plt.cm.nipy_spectral
        colors = [colormap(i) for i in np.linspace(0.3, 0.9, num_plots)]
        ax1.set_prop_cycle('color', colors)

        for i in range(num_plots):
            ax1.plot(slopes_l.index, slopes_l.iloc[:,i])
        ax1.plot(pchip.index, pchip, color = 'dimgrey')
        ax1.scatter(raw.index, raw, color='none', edgecolors="black", s=11)
        ax1.legend(slopes_l.columns.tolist() + ['Prior', 'Observations'], title = 'Total Data Percentage')
        for i in range(num_plots):
            ax1.plot(slopes_r.index, slopes_r.iloc[:,i])
        ax1.set_title('Prior')
        ax1.set_ylabel('Groundwater Level')
        ax1.text(x=0.05, y=-0.15, s = meta_l.to_string(index=True, float_format = "{0:.5}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes)
        ax1.text(x=-0.05, y=-0.08, s = 'Left Percentage',
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes)
        ax1.text(x=0.60, y=-0.15, s = meta_r.to_string(index=True, float_format = "{0:.5}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes)
        ax1.text(x=0.48, y=-0.08, s = 'Right Percentage',
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes)
        plt.tight_layout()
        fig.savefig(self.figures_root + '/' + str(name) + '_00_Trend' + extension)
        if show: plt.show(fig)
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()

    def rw_plot(self, y, rw, name, save = False, extension = '.png', show=False):
        fig = plt.figure(figsize=(12, 8))
        plt.plot(rw)
        plt.scatter(y.index, y, s=3, c = 'black')
        plt.ylabel('Groundwater Level')
        plt.legend(rw.columns.tolist() + ['Observations'])
        plt.title('Long-Term Trends: ' + str(name))
        if save: plt.savefig(self.figures_root + '/' + str(name) + '_00_RW' + extension)
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()
    
    def Q_Q_plot(self, Prediction, Observation, name, limit_low = 0, limit_high = 1, extension = '.png', show=False):
        #Plotting Prediction Correlation
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.scatter(Prediction, Observation)
        plt.ylabel('Observation')
        plt.xlabel('Prediction')
        plt.legend(['Prediction', 'Observation'])
        plt.title('Prediction Correlation: ' + name)
        cor_line_x = np.linspace(limit_low, limit_high, 9)
        cor_line_y = cor_line_x
        plt.xlim(limit_low, limit_high)
        plt.ylim(limit_low, limit_high)
        plt.plot(cor_line_x, cor_line_y, color='r')
        ax1.set_aspect('equal', adjustable='box')
        plt.savefig(self.figures_root + '/' + name + '_01_Q_Q' + extension)
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()

    def observeation_vs_prediction_plot(self, Prediction_X, Prediction_Y, Observation_X, Observation_Y, name, metrics=None, error_on = False, show=False):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(Prediction_X, Prediction_Y, "darkblue")
        # Potential bug # pandas version 1.4 pandas.errors.InvalidIndexError: (slice(None, None, None), None)
        ax.plot(Observation_X, Observation_Y, label= 'Observations', color='darkorange')
        ax.set_ylabel('Groundwater Level')
        ax.legend(['Prediction', 'Observation'])
        ax.set_title('Observation Vs Prediction: ' + name)
        if error_on:
          ax.text(x=0.0, y=-0.15, s = metrics[['Train ME','Train RMSE', 'Train MAE', 'Train r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
          ax.text(x=0.25, y=-0.15, s = metrics[['Validation ME','Validation RMSE', 'Validation MAE', 'Validation r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)          
          extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
          fig.savefig(self.figures_root  + '/' + name + '_02_Observation', bbox_inches=extent.expanded(1.1, 1.6))
        else: fig.savefig(self.figures_root  + '/' + name + '_02_Observation')
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()
        
    def residual_plot(self, Prediction_X, Prediction_Y, Observation_X, Observation_Y, name, show=False):
        date_rng = pd.DataFrame(np.arange(0, len(Observation_X), 1), index = Observation_X)
        data = self.Data_Join(Prediction_Y, Observation_Y).dropna()
        data = self.Data_Join(data, date_rng).dropna()
        data.columns = ['Prediction_Y', 'Observation_Y', 'Spline Index']
        residual = data['Prediction_Y'] - data['Observation_Y']
        spline = interpolate.UnivariateSpline(data['Spline Index'], residual, s=1000000)
        
        fig, axs = plt.subplots(2, 1, figsize = (12,12))
        axs[0].plot(data.index, residual, marker = 'o', linestyle='None', markersize=5, color = "black")
        axs[0].plot(Observation_X, np.zeros(shape = (len(Observation_X), 1)), color = 'royalblue', linewidth= 2.0)
        axs[0].plot(data.index, spline(data['Spline Index']), color = "red", linewidth= 3.0)
        axs[0].legend(['Residual', 'Zero', 'Spline Trend'])
        
        axs[0].set_ylabel('Prediction Residual Error')
        axs[0].set_title('Residual Error: ' + str(name))
        
        axs[1].plot(Observation_X, Observation_Y, marker = 'o', linestyle='None', markersize=5, color = "black")
        axs[1].plot(Observation_X, Observation_Y[name].mean()*np.ones(shape = (len(Observation_X), 1)), color = 'royalblue', linewidth= 2.0)
        axs[1].legend(['Observations', 'Mean'])
        axs[1].set_ylabel('Groundwater Level')
        axs[1].set_title('Groundwater Observations: ' + str(name))
        
        plt.savefig(self.figures_root  + '/' + str(name) + '_03_Residual_Plot')
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)  
        gc.collect()

    def observeation_vs_imputation_plot(self, Prediction_X, Prediction_Y, Observation_X, Observation_Y, name, show=False):
        fig = plt.figure(figsize=(12, 8))
        plt.plot(Prediction_X, Prediction_Y, "darkblue")
        plt.plot(Observation_X, Observation_Y, label= 'Observations', color='darkorange')
        plt.ylabel('Groundwater Level')
        plt.xlabel('Date')
        plt.legend(['Imputed Values', 'Smoothed Observations'])
        plt.title('Observation Vs Imputation: ' + name)
        plt.savefig(self.figures_root  + '/' + name + '_04_Imputation')
        if show: plt.show()
        else: 
            fig.clf()
            plt.close(fig)
        gc.collect()

    def raw_observation_vs_prediction(self, Prediction, Raw, name, Aquifer, metrics=None, error_on = False, test=False, show=False):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(Prediction.index, Prediction, 'darkblue', label='Prediction', linewidth=1.0)
        ax.scatter(Raw.index, Raw, color='darkorange', marker = '*', s=10, label= 'Observations')
        ax.set_title(Aquifer + ': ' + 'Well: ' + name + ' Predicted Values')
        ax.set_ylabel('Groundwater Level')
        ax.legend(fontsize = 'medium')
        if error_on:
          ax.text(x=0.0, y=-0.15, s = metrics[['Train ME','Train RMSE', 'Train MAE', 'Train r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
          ax.text(x=0.25, y=-0.15, s = metrics[['Validation ME','Validation RMSE', 'Validation MAE', 'Validation r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)          
          extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
          if test:
               ax.text(x=0.5, y=-0.15, s = metrics[['Test ME','Test RMSE', 'Test MAE', 'Test r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
          fig.savefig(self.figures_root  + '/' + name + '_05_Prediction_vs_Raw', bbox_inches=extent.expanded(1.3, 1.6))
        else: fig.savefig(self.figures_root  + '/' + name + '_05_Prediction_vs_Raw')
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()
        
    def raw_observation_vs_filled(self, Prediction, Raw, name, Aquifer, df_spread=None, conf_interval = None, 
                                  ci = 1, metrics=None, error_on = False, test=False, show=False):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(Prediction.index, Prediction, 'darkblue', label='Prediction', linewidth=1.0)
        ax.plot(Raw.index, Raw, color='darkorange', label= 'Observations', linewidth=1.0)
        ax.set_title(Aquifer + ': ' + 'Well: ' + name + ' Imputed Values')
        ax.set_ylabel('Groundwater Level')
        if conf_interval and not df_spread.empty:
            upper = df_spread['mean'] + df_spread['std'] * ci
            lower = df_spread['mean'] - df_spread['std'] * ci
            data_min = min(Prediction.min(), Raw.min())
            data_max = max(Prediction.max(), Raw.max())
            ax.fill_between(df_spread.index, upper, lower, color='blue', alpha=0.2, label=f'Std Dev: {ci}')
            ax.set_ylim(data_min - 5, data_max + 5)
        ax.legend(fontsize = 'medium')

        if error_on:
          ax.text(x=0.0, y=-0.15, s = metrics[['Train ME','Train RMSE', 'Train MAE', 'Train r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
          ax.text(x=0.25, y=-0.15, s = metrics[['Validation ME','Validation RMSE', 'Validation MAE', 'Validation r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)          
          extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
          if test:
               ax.text(x=0.5, y=-0.15, s = metrics[['Test ME','Test RMSE', 'Test MAE', 'Test r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
          fig.savefig(self.figures_root  + '/' + name + '_05_Filled_vs_Raw', bbox_inches=extent.expanded(1.3, 1.6))
        else: fig.savefig(self.figures_root  + '/' + name + '_05_Filled_vs_Raw')
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()
    
    def raw_observation_vs_imputation(self, Prediction, Raw, name, Aquifer, metrics=None, show=False):
        fig = plt.figure(figsize=(12, 8))
        plt.plot(Prediction.index, Prediction, 'darkblue', label='Model', linewidth=1.0)
        plt.scatter(Raw.index, Raw, color='darkorange', marker = '*', s=10, label= 'Observations')
        plt.title(Aquifer + ': ' + 'Well: ' + name + ' Raw vs Model')
        plt.legend(fontsize = 'medium')
        plt.savefig(self.figures_root  + '/' + name + '_06_Imputation_vs_Raw')
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()

    def observeation_vs_prediction_scatter_plot(self, Prediction, Y_train, Y_val, name, metrics=None, error_on = False, show=False):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(Prediction.index, Prediction, "darkblue", linewidth=1.0)
        ax.scatter(Y_train.index, Y_train, color='darkorange', marker='*', s=10)
        ax.scatter(Y_val.index, Y_val, color='lightgreen', s=10)  
        ax.legend(['Prediction', 'Training Data', 'Validation Data'])
        ax.set_ylabel('Groundwater Level')
        ax.set_title('Observation Vs Prediction: ' + name)
        if error_on:
          ax.text(x=0.0, y=-0.15, s = metrics[['Train ME','Train RMSE', 'Train MAE', 'Train r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
          ax.text(x=0.25, y=-0.15, s = metrics[['Validation ME','Validation RMSE', 'Validation MAE', 'Validation r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)          
          extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
          fig.savefig(self.figures_root  + '/' + name + '_07_Observation', bbox_inches=extent.expanded(1.2, 1.6))
        else: fig.savefig(self.figures_root  + '/' + name + '_07_Observation')
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()
    
    def prediction_vs_test(self, Prediction, Well_set_original, y_test, name, metrics=None, error_on = False, show=False):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(Prediction.index, Prediction, "darkblue", linewidth=1.0)
        ax.scatter(Well_set_original.index, Well_set_original, color='darkorange', marker='*', s=10)
        ax.scatter(y_test.index, y_test, color='lightgreen', s=10)
        ax.set_ylabel('Groundwater Level')
        ax.legend(['Prediction', 'Training Data', 'Test Data'])
        ax.set_title('Observation Vs Prediction: ' + name)
        ax.axvline(dt.datetime(int(self.cut_left), 1, 1), linewidth=0.25)
        ax.axvline(dt.datetime(int(self.cut_right), 1, 1), linewidth=0.25)
        if error_on:
          ax.text(x=0.0, y=-0.15, s = metrics[['Train ME','Train RMSE', 'Train MAE', 'Train r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
          ax.text(x=0.25, y=-0.15, s = metrics[['Validation ME','Validation RMSE', 'Validation MAE', 'Validation r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)  
          ax.text(x=0.5, y=-0.15, s = metrics[['Test ME','Test RMSE', 'Test MAE', 'Test r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
          extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
          fig.savefig(self.figures_root  + '/' + name + '_08_Test', bbox_inches=extent.expanded(1.2, 1.6))
        else: fig.savefig(self.figures_root  + '/' + name + '_08_Test')
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()
        
    def prediction_kfold(self, Prediction, Well_set_original, y_test, name, metrics=None, error_on = False, show=False, plot=False):
        if plot == True:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            ax.plot(Prediction.index, Prediction, "darkblue", linewidth=1.0)
            ax.scatter(Well_set_original.index, Well_set_original, color='darkorange', marker='*', s=10)
            ax.scatter(y_test.index, y_test, color='lightgreen', s=10)
            ax.set_ylabel('Groundwater Level')
            ax.legend(['Prediction', 'Training Data', 'Test Data'])
            ax.set_title('Observation Vs Prediction: ' + name)
            ax.axvline(y_test.index[0], linewidth=0.25)
            ax.axvline(y_test.index[-1], linewidth=0.25)
            if error_on:
              ax.text(x=0.0, y=-0.15, s = metrics[['Train ME','Train RMSE', 'Train MAE', 'Train r2']].to_string(index=True, float_format = "{0:.3}".format),
                      fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
              ax.text(x=0.25, y=-0.15, s = metrics[['Validation ME','Validation RMSE', 'Validation MAE', 'Validation r2']].to_string(index=True, float_format = "{0:.3}".format),
                      fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)  
              ax.text(x=0.5, y=-0.15, s = metrics[['Test ME','Test RMSE', 'Test MAE', 'Test r2']].to_string(index=True, float_format = "{0:.3}".format),
                      fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
              extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
              fig.savefig(self.figures_root  + '/' + name + '_08_kfold', bbox_inches=extent.expanded(1.2, 1.6))
            else: fig.savefig(self.figures_root  + '/' + name + '_08_kfold')
            if show: plt.show()
            else:
                fig.clf()
                plt.close(fig)
            gc.collect()
        
    def prediction_vs_test_kfold(self, Prediction, Well_set_original, name, metrics=None, error_on = False, show=False):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(Prediction.index, Prediction, "darkblue", linewidth=1.0)
        ax.scatter(Well_set_original.index, Well_set_original, color='darkorange', marker='*', s=10)
        ax.set_ylabel('Groundwater Level')
        ax.legend(['Prediction', 'Training Data', 'Test Data'])
        ax.set_title('Observation Vs Prediction: ' + name)
        if error_on:
          ax.text(x=0.0, y=-0.15, s = metrics[['Train ME','Train RMSE', 'Train MAE', 'Train r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
          ax.text(x=0.25, y=-0.15, s = metrics[['Validation ME','Validation RMSE', 'Validation MAE', 'Validation r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)  
          ax.text(x=0.5, y=-0.15, s = metrics[['Test ME','Test RMSE', 'Test MAE', 'Test r2']].to_string(index=True, float_format = "{0:.3}".format),
                  fontsize = 12, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
          extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
          fig.savefig(self.figures_root  + '/' + name + '_08_Test_kfold', bbox_inches=extent.expanded(1.2, 1.6))
        else: fig.savefig(self.figures_root  + '/' + name + '_08_Test_kfold')
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)    
        gc.collect()
        
    def Feature_Importance_box_plot(self, importance_df, show=False):
        #All Data       
        importance_df.boxplot(figsize=(20,10))
        plt.xlabel('Feature')
        plt.ylabel('Relative Importance')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(self.figures_root  + '/' + 'Feature_Importance_Complete')
        if show: plt.show()
        else: plt.close()
    
        #Calc Mean and sort
        importance_mean_df = importance_df.mean()
        importance_mean_df = pd.DataFrame(importance_mean_df.sort_values(axis=0, ascending=False)).T
        importance_mean = importance_df.transpose().reindex(list(importance_mean_df.columns)).transpose()
        importance_mean.iloc[:,:10].boxplot(figsize=(5,5))
        plt.xticks(rotation=90)
        plt.xlabel('Feature')
        plt.ylabel('Relative Importance')
        plt.title('Most Prevalent Features:')
        plt.tight_layout()
        plt.savefig(self.figures_root  + '/' + 'Feature_Importance_Uppper')
        if show: plt.show()
        else: plt.close()
        
        #Lower
        importance_mean.iloc[:,importance_mean.shape[1]-10:].boxplot(figsize=(5,5))
        plt.xticks(rotation=90)
        plt.xlabel('Feature')
        plt.ylabel('Relative Importance')
        plt.title('Least Prevalent Features:')
        plt.tight_layout()
        plt.savefig(self.figures_root  + '/' + 'Feature_Importance_Lower')
        if show: plt.show()
        else: plt.close()
        gc.collect()

    def feature_plot(self, Feature_Data, raw, name, show=False):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(Feature_Data)
        for i, n in enumerate(Feature_Data.columns[1:]):
            ax.scatter(raw.index, raw[n], color='none', edgecolors='black', s=10)
        legend = Feature_Data.columns.tolist() + ['Observed Measurements']
        ax.legend(legend, loc="lower left", bbox_to_anchor=(0.02, -0.25),
          ncol=2, fancybox=True, shadow=True)
        ax.set_ylabel('Groundwater Level')
        ax.set_title('Well Feature Correlation: ' + str(name))
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(self.figures_root  + '/' + str(name) + '_09_Features', bbox_inches=extent.expanded(1.3, 1.8))
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()

    def Aquifer_Plot(self, imputed_df, show=False):
        fig = plt.figure(figsize=(12, 8))
        plt.plot(imputed_df)
        plt.title('Measured and Interpolated data for all wells')
        plt.savefig(self.figures_root  + '/' + 'Aquifer_Plot')
        if show: plt.show()
        else:
            fig.clf()
            plt.close(fig)
        gc.collect()
           