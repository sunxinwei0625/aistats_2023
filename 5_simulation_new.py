#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from collections import defaultdict

pd.set_option('display.max_columns', 300)
pd.set_option('display.max_rows', 300)
pd.set_option('display.max_colwidth', 300)

# generate data
def get_data(n, parameters, seed=1234):
    """
    Parameters
    ----------
    n : int,sample size

    Returns
    -------
    data: pd.DataFrame, data

    """
    np.random.seed(seed)
    # generate T
    p_t = parameters['p_t']
    T=np.random.binomial(1,p_t,size=n)

    # generate X
    mu_x, sigma_x = parameters['mu_x'], parameters['sigma_x']
    X=np.random.normal(mu_x, sigma_x, size=n)

    # generate M1
    a = parameters['a']
    sigma_m1 = parameters['sigma_m1']
    M1 = a[0] + a[1]*T + a[2]*X + np.random.normal(0, sigma_m1, size=n)

    # generate M2
    b = parameters['b']
    sigma_m2 = parameters['sigma_m2']
    M2 = b[0] + b[1]*M1 + b[2]*T + b[3]*X + np.random.normal(0, sigma_m2, size=n)

    # generate Y
    rho = parameters['rho']
    sigma_y = parameters['sigma_y']
    Y = rho[0] + rho[1]*M1 + rho[2]*M2 + rho[3]*X + rho[4]*T + np.random.normal(loc=0, scale=sigma_y, size=n)
    data = pd.DataFrame(np.transpose(np.array((T,X,M1,M2,Y))),\
                        columns=['T','X','M1','M2','Y'])
    return data

def get_effect(parameters):
    """
    Parameters
    ----------
    parameters : dictionary, specifies the distribution of (T,X,M1,M2,Y)

    Returns
    -------
    mediation effects calculated by Monte Carlo:
        Delta_m1(t)
        Delta_m2(t)
    """
    a = parameters['a']
    b = parameters['b']
    rho = parameters['rho']

    effect = a[1]*(rho[1]+rho[2]*b[1]) + (rho[4]+rho[2]*b[2])
    delta_m1 = rho[1]*a[1]+rho[2]*b[1]*a[1]
    delta_m2 = rho[2]*b[1]*a[1]+rho[2]*b[2]
    delta_m1_m2 = rho[1]*a[1] + rho[2]*(b[1]*a[1]+b[2])
    zeta = rho[4]

    results = defaultdict(list)
    results['adj_M1'].append(delta_m1)
    results['adj_M2'].append(delta_m2)
    results['adj_M1M2'].append(delta_m1_m2)

    results['unadj_M1'].append(effect-delta_m1)
    results['unadj_M2'].append(effect-delta_m2)
    results['unadj_M1M2'].append(effect-delta_m1_m2)

    results['sample_size'].append(None)
    results['seed'].append(None)
    return results


def get_E_Y_model(data, model_y, var_y, vars_covariate, var_treat, need_cv=False, param_grid=None, seed=1234):
    '''
    Get the estimator of E[Y|x,m,r].
    vars_covariate is a list of str, including the names of mediators and confounders
    '''
    np.random.seed(seed)
    data_train  = data.loc[:, [var_y]+vars_covariate+[var_treat]].copy()

    dtypes_var = data_train.dtypes
    var_need_dummy = list(dtypes_var.index[dtypes_var=='object'])
    var_numeric = list(dtypes_var.index[dtypes_var!='object'])
    data_train_numeric= pd.concat([data.loc[:,var_numeric]]+[pd.get_dummies(data[var]) for var in var_need_dummy],axis=1).dropna()

    # predictors and target
    X=data_train_numeric.drop(var_y, axis=1)
    Y=data_train_numeric[var_y]

    vars_regression=X.columns

    if need_cv:
        # select hyper parameters by cross-validation
        grid_search = GridSearchCV(model_y, param_grid=param_grid, cv=5)
        grid_search.fit(X, Y) # find the best set of hyper-parameter
        model_y=grid_search.best_estimator_
    else:
        model_y.fit(X, Y)

    return model_y, vars_regression

def get_X(data, vars_continuous, vars_discrete):
    '''
    Get the grid points of X=vars_x_numeric+vars_x_discrete

    Parameters
    ----------
    data : pd.DataFrame// output of get_data()

    vars_continuous: list of str // continuousvariables that we caculate their density
    vars_discrete: list of str// discrete variables that we calculate the categorical probability

    Returns
    -------
    X : pd.DataFrame// columns = vars_continuous+vars_discrete

    '''
    if len(vars_continuous)+len(vars_discrete)>0:
        #lists_continuous = [np.arange(data[var].min(), data[var].max()) for var in vars_continuous]
        lists_continuous = [np.arange(-int(max(abs(data[var]))+1), int(max(abs(data[var]))+1)+1) for var in vars_continuous]

        lists_discrete = [data[var].dropna().unique() for var in vars_discrete]

        lists=lists_continuous + lists_discrete

        X_product = [[val] for val in lists[0]]
        for l in lists[1:]:
            X_product = [vals_old+[val_new] for vals_old in X_product for val_new in l]

        X=pd.DataFrame(X_product)
        X.columns =  vars_continuous+vars_discrete
    else:
        X=[]
    return X

def get_P_x_t_auto(data, var_treat, vars_continuous, vars_discrete, seed=1234):
    np.random.seed(seed)
    if len(vars_continuous)>0:
        # vars_continuous
        param_grid = {'bandwidth':list(np.arange(1,5,step=0.2))}
        grid_search = GridSearchCV(KernelDensity(), param_grid)
        p_continuous_0=grid_search.fit(data.loc[data[var_treat]==0, vars_continuous].dropna().values.reshape(-1,len(vars_continuous))).best_estimator_
        p_continuous_1=grid_search.fit(data.loc[data[var_treat]==1, vars_continuous].dropna().values.reshape(-1,len(vars_continuous))).best_estimator_

    if len(vars_discrete)>0:
        # vars_discrete
        models_dict = {}
        features_dict = {}
        for i in range(len(vars_discrete)):
            var = vars_discrete[i]
            vars_feature = vars_discrete[0:i]+vars_continuous+[var_treat]
            dtypes_feature = data.loc[:,vars_feature].dtypes
            feature_need_dummy = list(dtypes_feature.index[dtypes_feature=='object'])
            feature_numeric = list(dtypes_feature.index[dtypes_feature!='object'])
            data_numeric= pd.concat([data.loc[:,feature_numeric+[var]]]+\
                                             [pd.get_dummies(data[var]) for var in feature_need_dummy],axis=1)

            vars_feature_new = list(set(list(data_numeric.columns))-set([var]))
            features_dict[var] = vars_feature_new

            model = LogisticRegression(random_state=1234, max_iter=2000, penalty='l2')

            data_train = data_numeric.loc[:, vars_feature_new+[var]].dropna()
            data_x = data_train.loc[:, features_dict[var]]
            data_y = data_train.loc[:, var]

            model.fit(data_x, data_y)
            models_dict[var]=model
    # Given X calculate p(x|t=0) and p(x|t=1) for each row x
    def P_x_t(X, t):
        X_dtypes = X.dtypes
        vars_numeric =  list(X_dtypes.index[X_dtypes!='object'])
        vars_categoric = list(X_dtypes.index[X_dtypes=='object'])
        X_numeric = pd.concat([X.loc[:,vars_numeric]]+\
                              [pd.get_dummies(X[var]) for var in vars_categoric], axis=1)
        X_numeric[var_treat]=t
        # density of contiuous variables
        if len(vars_continuous)>0:
            if t==0:
                pr = np.exp(p_continuous_0.score_samples(X[vars_continuous].values))
            else:
                pr = np.exp(p_continuous_1.score_samples(X[vars_continuous].values))
        else:
            pr=1
        # times the probability of discrete variables
        if len(vars_discrete)>0:
            X_temp = X.copy()
            X_temp['treat'] = t
            X_temp['index'] = X_temp.index
            for var in vars_discrete:
                vars_feature = features_dict[var]
                model = models_dict[var]
                probs_predict = pd.DataFrame(model.predict_proba(X_numeric.loc[:,vars_feature]),\
                                             columns=model.classes_)
                probs_predict['index'] = X_temp['index']
                pr_var = X_temp.apply(lambda x: probs_predict.loc[x['index'],x[var]], axis=1)
                pr *= pr_var
        return pr
    return P_x_t

def change_to_numeric(M):
    M_dtypes = M.dtypes
    vars_m_numeric =  list(M_dtypes.index[M_dtypes!='object'])
    vars_m_categoric = list(M_dtypes.index[M_dtypes=='object'])
    M_numeric = pd.concat([M.loc[:,vars_m_numeric]]+\
                          [pd.get_dummies(M[var]) for var in vars_m_categoric], axis=1)
    return M_numeric


def get_Counterfactual_Mean(data, X, prob_x_t, M, prob_m_t, E_Y_model, vars_regression, var_treat):
    '''
    Get E[Y(r,M(1))], E[Y(r,M(0))] based on E[Y(r, Y(1/0))]=\sum_{m,x}{E[Y|m,r,x]p(x|r)p(m|R=1/0)}

    Returns
    -------
    results : Tuple, (E[Y(1,M(1))], E[Y(1,M(0))], E[Y(0,M(1))], E[Y(0,M(0))])
    '''
    # calculate E[Y(0,M(0))], E[Y(0,M(1))], E[Y(1,M(0))], E[Y(1,M(1))]
    # e.g. E_Y_0_M_0 = \sum_x\sum_m E[Y|x,m,0] p(m|r)
    E_Y_0_M_0, E_Y_0_M_1 = 0, 0
    E_Y_1_M_0, E_Y_1_M_1 = 0, 0

    # If X.shape[0]<M.shape[0], Iterate on nrow_x
    if X.shape[0]<=M.shape[0]:
        # t0 = time.time()
        # print('time_start=0')
        nrow_x = X.shape[0]
        vars_x = X.columns
        M = change_to_numeric(M)
        for i in range(nrow_x):
            # t = time.time()
            # if i%100==0: print('i={} time={}'.format(i, t-t0))
            XMT = M.copy()
            x=X.iloc[i,:]
            for var in vars_x:
                XMT[var]=x[var]
            # E_Y_0_M_0 = E[Y(0, M(0))]
            XMT[var_treat]=0
            Y_fit = E_Y_model.predict(XMT.loc[:,vars_regression])
            p_x_t = prob_x_t[0][i]
            E_Y_0_M_0 += (Y_fit * p_x_t * prob_m_t[0]).sum()

            # E_Y_0_M_1 = E[Y(0, M(1))]
            XMT[var_treat]=0
            Y_fit = E_Y_model.predict(XMT.loc[:,vars_regression])
            p_x_t = prob_x_t[0][i]
            E_Y_0_M_1 += (Y_fit * p_x_t * prob_m_t[1]).sum()

            # E_Y_1_M_0 = E[Y(1, M(0))]
            XMT[var_treat]=1
            Y_fit = E_Y_model.predict(XMT.loc[:,vars_regression])
            p_x_t = prob_x_t[1][i]
            E_Y_1_M_0 += (Y_fit * p_x_t * prob_m_t[0]).sum()

            # E_Y_1_M_1 = E[Y(1, M(1))]
            XMT[var_treat]=1
            Y_fit = E_Y_model.predict(XMT.loc[:,vars_regression])
            p_x_t = prob_x_t[1][i]
            E_Y_1_M_1 += (Y_fit * p_x_t * prob_m_t[1]).sum()
    elif M.shape[0]<X.shape[0]:
        # t0 = time.time()
        # print('time_start=0')
        M = change_to_numeric(M)
        nrow_m = M.shape[0]
        vars_m = M.columns
        for i in range(nrow_m):
            # t = time.time()
            # if i==nrow_m-1: print('i={} time={}'.format(i, t-t0))
            XMT = X.copy()
            m=M.iloc[i,:]
            for var in vars_m:
                XMT[var]=m[var]

            # E_Y_0_M_0 = E[Y(0, M(0))]
            XMT[var_treat]=0
            Y_fit = E_Y_model.predict(XMT.loc[:,vars_regression])
            p_m_t = prob_m_t[0][i]
            E_Y_0_M_0 += (Y_fit * prob_x_t[0] * p_m_t).sum()

            # E_Y_0_M_1 = E[Y(0, M(1))]
            XMT[var_treat]=0
            Y_fit = E_Y_model.predict(XMT.loc[:,vars_regression])
            p_m_t = prob_m_t[1][i]
            E_Y_0_M_1 += (Y_fit * prob_x_t[0] * p_m_t).sum()

            # E_Y_1_M_0 = E[Y(1, M(0))]
            XMT[var_treat]=1
            Y_fit = E_Y_model.predict(XMT.loc[:,vars_regression])
            p_m_t = prob_m_t[0][i]
            E_Y_1_M_0 += (Y_fit * prob_x_t[1] * p_m_t).sum()

            # E_Y_1_M_1 = E[Y(1, M(1))]
            XMT[var_treat]=1
            Y_fit = E_Y_model.predict(XMT.loc[:,vars_regression])
            p_m_t = prob_m_t[1][i]
            E_Y_1_M_1 += (Y_fit * prob_x_t[1] * p_m_t).sum()

            #print('i={} time={} Counterfactual={}'.format(i, t-t0, [[E_Y_0_M_0, E_Y_0_M_1],[E_Y_1_M_0, E_Y_1_M_1]]))
    return [[E_Y_0_M_0, E_Y_0_M_1],[E_Y_1_M_0, E_Y_1_M_1]]



    return [[E_Y_0_M_0, E_Y_0_M_1],[E_Y_1_M_0, E_Y_1_M_1]]

def save_add_results(results, E_Y_t_M_t, data, var_y, var_treat, n, seed):
    results['sample_size'].append(n)
    results['mediation(1)'].append(E_Y_t_M_t[1][1]-E_Y_t_M_t[1][0])
    results['mediation(0)'].append(E_Y_t_M_t[0][1]-E_Y_t_M_t[0][0])
    results['direct(1)'].append(E_Y_t_M_t[1][1]-E_Y_t_M_t[0][1])
    results['direct(0)'].append(E_Y_t_M_t[1][0]-E_Y_t_M_t[0][0])
    results['seed'].append(seed)
    # results['total'].append(data.loc[data[var_treat]==1, var_y].mean()-data.loc[data[var_treat]==0, var_y].mean())
    return results



# n, parameters: for generate data
# need_cv, param_grid: training the regressor Y
# def get_mediation_result(n, parameters, model_y, \
#                          var_treat, vars_m_continuous, vars_m_discrete, vars_x_continuous, vars_x_discrete,\
#                          seed=1234, results=defaultdict(list), need_cv=False, param_grid=None):
#     ####### get data with sample_size=n ####################################
#     data = get_data(n, parameters, seed=seed)

#     ####### the grid value of X ############################################
#     X = get_X(data, vars_x_continuous, vars_x_discrete)


#     ####### get p(x|t=0) and p(x|t=1) by automatic factorizations ##########
#     p_x_t_model = get_P_x_t_auto(data, var_treat, vars_x_continuous, vars_x_discrete, seed=seed)
#     X=get_X(data, vars_x_continuous, vars_x_discrete)
#     prob_x_t = [p_x_t_model(X,0), p_x_t_model(X,1)]


#     ####### get p(m|t=0) and p(x|t=1) ##############
#     p_m_t_model = get_P_x_t_auto(data, var_treat, vars_m_continuous, vars_m_discrete, seed=seed)
#     M = get_X(data, vars_m_continuous, vars_m_discrete)
#     prob_m_t = [p_m_t_model(M,0), p_m_t_model(M,1)]

#     ####### get E[Y|x,m,r] ##############
#     vars_covariate = vars_x_discrete+vars_x_continuous+vars_m_continuous+vars_m_discrete
#     # need_cv = True
#     # param_grid = {'max_depth': [2, 5],'n_estimators': [50, 200, 500],'min_samples_leaf': [5, 10, 20]}
#     E_Y_model, vars_regression = get_E_Y_model(data, model_y, var_y, vars_covariate, var_treat, seed=seed) #need_cv, param_grid)

#     ####### get Counterfactual E[Y(r, M(r))] ##############
#     E_Y_t_M_t = get_Counterfactual_Mean(data, X, prob_x_t, M, prob_m_t, E_Y_model, vars_regression, var_treat)
#     results = save_add_results(results, E_Y_t_M_t, data, var_y, var_treat, n, seed)
#     return results



def get_mediation_effect_2(n, parameters, seed):
    '''
    specially for this case
    '''
    ####### get data with sample_size=n ####################################
    data = get_data(n, parameters, seed=seed)
    data['T1']=1
    data['T0']=0
    
    effect = float(data.loc[data['T']==1,['Y']].mean()-data.loc[data['T']==0,['Y']].mean())
    ####### calculate the counterfactural outcome M1(1) M1(0) M2(1) M2(0) ####################################
    model_M1 = LinearRegression()
    model_M1.fit(X=data.loc[:,['T', 'X']], y=data['M1'])
    # check
    # model_M1.coef_
    # a
    # check
    data['M1_1'] = model_M1.predict(data.loc[:,['T1', 'X']])
    data['M1_0'] = model_M1.predict(data.loc[:,['T0', 'X']])


    model_M2 = LinearRegression()
    model_M2.fit(X=data.loc[:,['M1','T','X']], y=data['M2'])
    data['M2_1'] = model_M2.predict(data.loc[:,['M1_1','T1', 'X']])
    data['M2_0'] = model_M2.predict(data.loc[:,['M1_0','T0', 'X']])

    ####### get Delta_M1(t) ##############

    model_y_3 = LinearRegression()
    vars_covariate = ['M1','T','X']
    model_y_3.fit(data.loc[:,vars_covariate], data['Y'])
    # Delta_M1(1) and Delta_M1(0)
    Delta_M1 = model_y_3.predict(data.loc[data['T']==0,['M1_1','T','X']]).mean() - \
               model_y_3.predict(data.loc[data['T']==0,['M1_0','T','X']]).mean()


    ####### get Delta_M2(t) ##############
    model_y_4 = LinearRegression()
    vars_covariate = ['M1','M2','X','T']
    model_y_4.fit(data.loc[:,vars_covariate], data['Y'])

    # Delta_M2(0) and Delta_M2(0)
    Delta_M2 = model_y_4.predict(data.loc[data['T']==0,['M1','M2_1','X','T']]).mean() - \
               model_y_4.predict(data.loc[data['T']==0,['M1','M2_0','X','T']]).mean()
    

    ####### get Delta_M1M2(t) ##############
    # Delta_M2(0) and Delta_M2(0)
    Delta_M1M2 = model_y_4.predict(data.loc[data['T']==0,['M1_1','M2_1','X','T']]).mean() - \
                   model_y_4.predict(data.loc[data['T']==0,['M1_0','M2_0','X','T']]).mean()

    ####### get zeta(t) ##############
    Zeta_M1 = effect - Delta_M1
    Zeta_M2 = effect - Delta_M2
    Zeta_M1M2 = effect - Delta_M1M2

    return {'delta_m1': Delta_M1, 'delta_m2': Delta_M2, 'delta_m1m2': Delta_M1M2,  \
            'zeta_m1': Zeta_M1, 'zeta_m2': Zeta_M2, 'zeta_m1m2': Zeta_M1M2,\
            'sample_size': n, 'seed': seed}



def save_add_results_2(results, results_add):
    '''
    results_add is the output of get_mediation_effect_2()
    '''
    results['adj_M1'].append(results_add['delta_m1'])
    results['adj_M2'].append(results_add['delta_m2'])
    results['adj_M1M2'].append(results_add['delta_m1m2'])

    results['unadj_M1'].append(results_add['zeta_m1'])
    results['unadj_M2'].append(results_add['zeta_m2'])
    results['unadj_M1M2'].append(results_add['zeta_m1m2'])

    results['sample_size'].append(results_add['sample_size'])
    results['seed'].append(results_add['seed'])
    return results











if __name__ == '__main__':
    parameters = {
        'p_t':0.5,
        'mu_x':1, 'sigma_x':1,
        'a': [0, 2, 3],
        'b': [0, 2, 3, 4],
        'rho': [0, 2, 3, 4, 5],
        'sigma_m1': 0.1, 'sigma_m2': 0.1, 'sigma_y':0.1
        }
    np.random.seed(123)
    seed_list = np.random.randint(100, 1000, size=100)
    sample_size = [500,1000,2000,5000]


    results = get_effect(parameters)

    np.random.seed(123)
    seed_list = np.random.randint(100, 1000, size=100)
    sample_size = [100, 500, 1000, 2000, 5000]

    for seed in seed_list:
        for n in sample_size:
            results_add = get_mediation_effect_2(n, parameters, seed)
            results = save_add_results_2(results, results_add)

    results = pd.DataFrame(results)
    results.to_csv('/Users/Documents/results_new.csv', index=False)

    # ####### alternative Method ##############
    # results_true = get_effect(parameters)

    # np.random.seed(123)
    # seed_list = np.random.randint(100, 1000, size=100)
    # sample_size = [500,1000,2000,5000]


    # results_M1=defaultdict(list)
    # for n in sample_size:
    #     for seed in seed_list:
    #         var_y, var_treat = 'Y', 'T'
    #         vars_m, vars_m_continuous, vars_m_discrete = [], ['M1'], []
    #         vars_x, vars_x_continuous, vars_x_discrete  = [], ['X'], []
    #         model_y = LinearRegression()
    #         results_M1 = get_mediation_result(n, parameters, model_y, var_treat, vars_m_continuous, \
    #                               vars_m_discrete, vars_x_continuous, vars_x_discrete, results=results_M1, seed=seed)
    # results_M1=pd.DataFrame(results_M1).loc[:,['sample_size','seed','mediation(1)','mediation(0)']]
    # results_M1.columns=['sample_size','seed', 'mediation(1)_M1', 'mediation(0)_M1']





    # results_M2=defaultdict(list)
    # for n in sample_size:
    #     for seed in seed_list:
    #         var_y, var_treat = 'Y', 'T'
    #         vars_m, vars_m_continuous, vars_m_discrete = [], ['M2'], []
    #         vars_x, vars_x_continuous, vars_x_discrete  = ['X','M1'], ['X','M1'], []
    #         model_y = LinearRegression()
    #         results_M2 = get_mediation_result(n, parameters, model_y, var_treat, vars_m_continuous, \
    #                               vars_m_discrete, vars_x_continuous, vars_x_discrete, results=results_M2, seed=seed)
    # results_M2=pd.DataFrame(results_M2).loc[:,['sample_size','seed','mediation(1)','mediation(0)']]
    # results_M2.columns= ['sample_size','seed', 'mediation(1)_M2', 'mediation(0)_M2']





    # results_M1M2=defaultdict(list)
    # for n in sample_size:
    #     for seed in seed_list:
    #         var_y, var_treat = 'Y', 'T'
    #         vars_m, vars_m_continuous, vars_m_discrete = [], ['M1','M2'], []
    #         vars_x, vars_x_continuous, vars_x_discrete  = ['X'], ['X'], []
    #         model_y = LinearRegression()
    #         results_M1M2 = get_mediation_result(n, parameters, model_y, var_treat, vars_m_continuous, \
    #                              vars_m_discrete, vars_x_continuous, vars_x_discrete, results=results_M1M2, seed=seed)
    # results_M1M2=pd.DataFrame(results_M1M2).loc[:,['sample_size','seed','mediation(1)','mediation(0)','direct(1)','direct(0)']]
    # results_M1M2.columns= ['sample_size','seed', 'mediation(1)_M1M2', 'mediation(0)_M1M2','direct(1)','direct(0)']
   



    # results_all = pd.concat([results_M1.loc[:,['sample_size','seed','mediation(1)','mediation(0)']],\
    #                          results_M2.loc[:,['mediation(1)','mediation(0)']],\
    #                          results_M1M2.loc[:,['mediation(1)','mediation(0)','direct(1)', 'direct(0)']]],axis=1)

    # results_append = pd.DataFrame([[float('inf'), None,results_true['mediation(1)_M1'][0], results_true['mediation(0)_M1'][0], \
    #                                           results_true['mediation(1)_M2'][0], results_true['mediation(0)_M2'][0], \
    #                                           results_true['mediation(1)_M1M2'][0], results_true['mediation(0)_M1M2'][0], \
    #                                           results_true['direct(1)'][0], results_true['direct(0)'][0]]], columns=results_all.columns)
    # results_all = results_all.append(results_append)


























# def get_mediation_effect_2(n, parameters, seed):
#     '''
#     specially for this case

#     '''
#     n=6000
#     ####### get data with sample_size=n ####################################
#     data = get_data(n, parameters, seed=seed)
#     data['T1']=1
#     data['T0']=0

#     ####### calculate the counterfactural outcome M1(1) M1(0) M2(1) M2(0) ####################################
#     model_M1 = LinearRegression()
#     model_M1.fit(X=data.loc[:,['T', 'X']], y=data['M1'])
#     data['M1_1'] = model_M1.predict(data.loc[:,['T1', 'X']])
#     data['M1_0'] = model_M1.predict(data.loc[:,['T0', 'X']])


#     model_M2 = LinearRegression()
#     model_M2.fit(X=data.loc[:,['M1','T','X']], y=data['M2'])
#     data['M2_1'] = model_M2.predict(data.loc[:,['M1_1','T1', 'X']])
#     data['M2_0'] = model_M2.predict(data.loc[:,['M1_0','T0', 'X']])

#     model_M2.coef_

#     ####### get E[Y|x,m,r] ##############
#     model_y = LinearRegression()
#     vars_covariate = vars_x_discrete+vars_x_continuous+vars_m_continuous+vars_m_discrete
#     E_Y_model, vars_regression = get_E_Y_model(data, model_y, var_y, vars_covariate, var_treat, seed=seed) #need_cv, param_grid)


#     ####### get Delta_M1(t) ##############
#     # check
#     # a, b, rho = parameters['a'], parameters['b'], parameters['rho']
#     # para1, para2, para3 = rho[1]+rho[2]*b[1], rho[4]+rho[2]*b[2], rho[3]+rho[2]*b[3]
#     # model_y_3 = LinearRegression()
#     # vars_covariate = ['M1','T','X']
#     # model_y_3.fit(data.loc[:,vars_covariate], data['Y'])
#     # data.loc[data['T']==1,['M1_1','T1','X']].apply(lambda x:x.mean(), axis=0)
#     # check

#     model_y_3 = LinearRegression()
#     vars_covariate = ['M1','T','X']
#     model_y_3.fit(data.loc[:,vars_covariate], data['Y'])
#     # Delta_M1(1) and Delta_M1(0)
#     Delta_M1_1 = model_y_3.predict(data.loc[data['T']==1,['M1_1','T1','X']]).mean() - \
#                  model_y_3.predict(data.loc[data['T']==1,['M1_0','T1','X']]).mean()
#     Delta_M1_0 = model_y_3.predict(data.loc[data['T']==0,['M1_1','T1','X']]).mean() - \
#                  model_y_3.predict(data.loc[data['T']==0,['M1_0','T1','X']]).mean()



#     ####### get Delta_M2(t) ##############
#     # check
#     # a, b, rho = parameters['a'], parameters['b'], parameters['rho']
#     # m2_gap = b[1]*a[1]+b[2]
#     # check
#     model_y_4 = LinearRegression()
#     vars_covariate = ['M1','M2','X','T']
#     model_y_4.fit(data.loc[:,vars_covariate], data['Y'])


#     # Delta_M2(0) and Delta_M2(0)
#     Delta_M2_1 = model_y_4.predict(data.loc[data['T']==1,['M1','M2_1','X','T1']]).mean() - \
#                  model_y_4.predict(data.loc[data['T']==1,['M1','M2_0','X','T1']]).mean()
#     Delta_M2_0 = model_y_4.predict(data.loc[data['T']==0,['M1','M2_1','X','T1']]).mean() - \
#                  model_y_4.predict(data.loc[data['T']==0,['M1','M2_0','X','T1']]).mean()



#     ####### get Delta_M1M2(t) ##############

#     # Delta_M2(0) and Delta_M2(0)
#     Delta_M1M2_1 = model_y_4.predict(data.loc[data['T']==1,['M1_1','M2_1','X','T1']]).mean() - \
#                  model_y_4.predict(data.loc[data['T']==1,['M1_0','M2_0','X','T1']]).mean()
#     Delta_M1M2_0 = model_y_4.predict(data.loc[data['T']==0,['M1_1','M2_1','X','T1']]).mean() - \
#                  model_y_4.predict(data.loc[data['T']==0,['M1_0','M2_0','X','T1']]).mean()

#     ####### get zeta(t) ##############
#     Zeta_1 = model_y_4.predict(data.loc[data['T']==1,['M1_1','M2_1','X','T']]).mean() - \
#                  model_y_4.predict(data.loc[data['T']==0,['M1_1','M2_1','X','T']]).mean()
#     Zeta_0 = model_y_4.predict(data.loc[data['T']==1,['M1_0','M2_0','X','T']]).mean() - \
#                  model_y_4.predict(data.loc[data['T']==0,['M1_0','M2_0','X','T']]).mean()


#     return {'delta_m1_0': Delta_M1_0, 'delta_m1_1': Delta_M1_1, \
#             'delta_m2_0': Delta_M2_0, 'delta_m2_1': Delta_M2_1,  \
#             'delta_m1_m2_0': Delta_M1M2_0, 'delta_m1_m2_1': Delta_M1M2_1,\
#             'zeta_0': Zeta_0, 'zeta_1': Zeta_1}



