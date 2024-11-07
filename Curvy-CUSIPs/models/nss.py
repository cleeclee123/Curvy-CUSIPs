import numpy as np
from scipy import optimize
from scipy.optimize import minimize
from scipy.optimize import check_grad
import pandas as pd
import random



"""
References:
https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/2738606/Wahlstr%C3%B8m2021_Article_AComparativeAnalysisOfParsimon.pdf?sequence=2
https://www.r-bloggers.com/2022/07/nelson-siegel-svensson-yield-curve-model-using-r-code/
https://deliverypdf.ssrn.com/delivery.php?ID=713094097085069095026119002078073069000039039014031001095005110028031090091006089071118026012125037127020071113073102127003097023054032039051005095126102077103088068084044104004103088089007127098090104125103094011087076120066067100084122003123097082&EXT=pdf&INDEX=TRUE
https://docs.scipy.org/doc/scipy/tutorial/optimize.html#sequential-least-squares-programming-slsqp-algorithm-method-slsqp
"""


class ZeroCouponCurve:
    
    """
    Estimate a zero coupon yield curve with the Nelson Siegel Svensson model
    
    """
    
    def __init__(self, yield_data, ytm_data, init_params, method = None,
                 fallback = None, percent = True, bounds = None, constraints = None):
        """
        Parameters
        ----------
        yield_data : DataFrame
            dataframe containing daily yield data
        ytm_data : DataFrame
            dataframe containing daily years to maturity data for each yield data point
            
        init_params : list
            list of initial parameter values
            
        percent : boolean
            if yield data are given in percent (ie. 3.51 for example instead of 0.0351) then True, otherwise False
            standard is True
            
        bounds : list
            bounds for parameter values in the optimization. list of pairwise values for each parameter
            if bounds = None, a standard set of parameters will be applied
        method : string
            main optimization problem to be used, default is SLSQP
        fallback : string
            alternative optimization method to be used if main method fails
        constraints : dict
            constraints for the optimization problem
        """
        
        self.yield_data = yield_data
        self.dates = self.yield_data['date']
        self.yield_data = self.yield_data.drop(columns = 'date')
        
        #Multiply ny 100 if yield data is not specified as percentage
        if percent == False:
            self.yield_data = np.array(self.yield_data) * 100
        else:
            self.yield_data = np.array(self.yield_data)
            
        
        self.ytm_data = ytm_data
        self.ytm_data = self.ytm_data.drop(columns = 'date')
        self.ytm_data = np.array(self.ytm_data)
        
        self.yield_data = self.yield_data.astype(float)
        self.ytm_data = self.ytm_data .astype(float)
        
        self.init_params = init_params
        
        if bounds == None:
            self.bounds = [(0,15),(-15,30),(-30,30),(-30,30),(0.001,30),(0.001,30)]
        else:
            self.bounds = bounds
            
        if constraints == None:
            self.cons=( 
                    {'type': 'ineq','fun': lambda x: x[0] + x[1]},
                    {'type': 'ineq','fun': lambda x: x[4]},
                    {'type': 'ineq','fun': lambda x: x[5]},
            )
        else:
            self.cons = constraints
        
        #If no optimization method is specified, use SLSQP as default
        if method == None:
            self.method = 'SLSQP'
        else:
            self.method = method
            
        #If no fallback optimization method is specified, use Nelder-Mead as default
        if fallback == None:
            self.fallback = 'Nelder-Mead'
        else:
            self.method = method

    def nss_model(self, T, params):
        """
        Return a yield for a given time to maturity (T) in years and a set of parameters
        """
        
        beta0 = params[0]
        beta1 = params[1]
        beta2 = params[2]
        beta3 = params[3]
        lambda0 = params[4]
        lambda1 = params[5]
        
        alpha1 = (1 - np.exp(-T / lambda0)) / (T / lambda0)
        alpha2 = (1 - np.exp(-T / lambda0)) / (T / lambda0) - np.exp(-T / lambda0)
        alpha3 = (1 - np.exp(-T / lambda1)) / (T / lambda1) - np.exp(-T / lambda1)
        
        return (beta0 + beta1 * alpha1 + beta2 * alpha2 + beta3 * alpha3)
    
    def nss_residuals(self, params, time_vec, yield_vec):
        """
        Parameters
        ----------
        params : list
            list of NSS parameters
        time_vec : list
            list of time to maturity values
        yield_vec : list
            list of yields corresponding to the given yield to maturity values
        """
        residuals = np.sum((self.nss_model(time_vec, params) - yield_vec)**2)
    
        return residuals
    

        
    def nss_optimize(self, params, time_vec, yield_vec, opt_method):
        """
        Parameters
        ----------
        params : numpy array
            array of initial parameter values
        time_vec : numpy array
            array of initial parameter values
        yield_vec : numpy array
            array of initial parameter values
        opt_method : string, optional
            set the optimization method. 
            The default is 'SLSQP'.
        """
        

        try:
            solution = minimize(self.nss_residuals, x0 = np.array(params), 
                                args = (time_vec, yield_vec), method = opt_method,
                                bounds = self.bounds,
                                constraints = self.cons)
            if solution.success:
                return solution.x, solution.fun
            else:
                return []
    
        except:
            pass
        
    def estimate_nss_params(self, mat_end, param_iter = 1):
        """
        estimate nss parameters for the whole dataset
        
        Parameters
        ----------
        mat_end : int
            last maturity to be estimated in months
        param_iter : int, optional
            Number of random trials for the optimization. If this is greater than one, the estimation process will
            select a random set of parameters and rerun the optimization. A the end, the parameter set
            with lowest residuals will be selected as the optimal set of parameters. The default is 1.

        """

        init_params = self.init_params
        results = np.arange(3,mat_end,1) / 12
        
        zc_rates = np.zeros((len(self.yield_data), len(results)))
        param_arr = np.zeros((len(self.yield_data), len(init_params)))
        resid_arr = []
        
        yields = self.yield_data
        ytm = self.ytm_data

        for i in range(0,len(yields)):
            print(i)
            y_vec = yields[i]
            t_vec = ytm[i]

            vec = np.stack((t_vec,y_vec)).T
            vec = vec[~np.isnan(vec[:,0])]
            vec = vec[~np.isnan(vec[:,1])]

            #Skip date if number of data points is lower than 4     
            if len(vec) < 4:
                pass
            else:
                vec = vec[vec[:,0].argsort()]
                
                #Adjust beta0 and beta1 to new set of yields
                beta0 = (vec[len(vec) - 1,1] + vec[len(vec) - 2,1]) / 2.0
                beta1 = vec[0][1] - beta0

                if len(init_params) > 0:
                    
                    init_params[0] = beta0
                    init_params[1] = beta1
                #If list of parameters from previous estimation is empty, choose the original initial parameters
                elif len(init_params) == 0:
                    init_params = self.init_params
                
                
                #Run through parameter iterations and get lists of residuals and tried parameters
                residuals, param_trials = self.run_param_iter(param_iter, init_params, vec)
                #Drop if residuals are zero
                residuals = residuals[~np.all(residuals == 0, axis = 1)]
                param_trials = param_trials[~np.all(param_trials == 0, axis = 1)]

                #Check if parameter list has any elements, skip if not
                if len(residuals) == 0:
                    pass
                else:
                    #Select the set of parameters with the lowest possible values
                    optim_params = param_trials[residuals.argmin()]
                    
                
                #Calculate a set of yields for the given parameters values
                res = self.nss_model(results, optim_params)
                param_arr[i] = optim_params
                zc_rates[i] = res
                resid_arr.append(residuals)
                
                #Set new parameter values to previously calculated optimal parameters
                init_params = optim_params
                
        return param_arr, zc_rates, resid_arr, vec
        
    def run_param_iter(self, param_iter, init_params, vec):
        
        residuals = np.zeros((param_iter,1))
        param_trials = np.zeros((param_iter, 6))
        param_trials[0] = init_params
        
        for j in range(param_iter):
            params = self.draw_new_init_params(init_params)
            #For the first estimation, use initial parameters init_params
            if j == 0:
                opt_res = self.nss_optimize(init_params, vec[:,0], vec[:,1], opt_method = self.method)
                #If minimize fails for the standard SLSQP method, try Nelder-Mead as a fallback
                if opt_res == None or len(opt_res) == 0:
                    print('Fallback-method used')
                    opt_res = self.nss_optimize(init_params, vec[:,0], vec[:,1], opt_method = self.fallback)
            #For subsequent parameter trials, use a randomly drawn parameter set
            else:
                opt_res = self.nss_optimize(params, vec[:,0], vec[:,1], opt_method = self.method)
                #If minimize fails for the standard SLSQP method, try Nelder-Mead as a fallback
                if opt_res == None or len(opt_res) == 0:
                    print('Fallback-method used')
                    opt_res = self.nss_optimize(init_params, vec[:,0], vec[:,1], opt_method = self.fallback)
            
            #If optimization still fails, then revert to the initial parameter set
            if opt_res == None or len(opt_res) == 0:
                print('Standard and fallback method failed, revert to initial parameter set')
                param_trials[j] = init_params
            
            else:
                param_trials[j] = opt_res[0]
                
                if opt_res[1] == 0:     #Set residuals to a very high value if they are zero. Zero residuals indicates a failed estimation
                    residuals[j] = 1000.0
                    print(residuals[j])
                else:
                            residuals[j] = opt_res[1]
                            
        return residuals, param_trials
                            
    def calc_nss_yields(self, parameters, results):
        
        yields = np.zeros((len(parameters), len(results)))
        
        for i in range(len(parameters)):
            yields[i] = self.nss_model(results, parameters[i])
            
        yields = pd.DataFrame(yields)
        yields = yields.loc[yields.sum(axis = 1) != 0]
        yields = pd.merge(self.dates, yields, left_on = self.dates.index, right_on = yields.index)
        yields = yields.drop(columns = 'key_0')
            
        return yields


    def draw_new_init_params(self, params):
        """
        Draw new initial parameters by randomly selecting another set of parameters
        beta0 and beta1 are selected as original values, but with a random adjustment
        """
        beta0 = params[0] + random.randint(-2,2)
        beta1 = params[1] + random.randint(-2,2)
        beta2 = random.randint(-30,30)
        beta3 = random.randint(-30,30)
        lambda0 = random.randint(0,30)
        lambda1 = random.randint(0,30)
        return [beta0, beta1, beta2, beta3, lambda0, lambda1]