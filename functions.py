#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:18:47 2023

@author: alejomonbar
"""
from docplex.mp.model import Model
import numpy as np
from openqaoa.workflows.optimizer import QAOA
# from joblib import Parallel, delayed
# import multiprocessing
import itertools
from copy import copy
from joblib import Parallel, delayed
import multiprocessing
from openqaoa.problems.converters import FromDocplex2IsingModel


num_cores = multiprocessing.cpu_count()

def portfolio_optimization(mu, sigma, risk, budget=None, min_profit=None):
   
    mdl = Model('Portfolio Optimization')
    N_stocks = len(mu)
    # Binary set of variables that represent the stocks
    x = np.array(mdl.binary_var_list(N_stocks, name="stock")) # x vector in numpy array for matrix multiplication
    
    # Portfolio optimization function
    objective_function = (mu @ x - risk * x.T @ sigma @ x) / np.max(np.abs(mu))
    mdl.maximize(objective_function)
    
    # Budget constraint
    if budget:
        mdl.add_constraint(mdl.sum(x) <= budget, ctname='budget')
    # Minimum profit
    if min_profit:
        mdl.add_constraint(mdl.sum(mui * xi / min_profit for mui, xi in zip(mu, x)) >= 1)
    return mdl

def solution_str(mdl):
    docplex_sol = mdl.solve()
    solution = ""
    for ii in mdl.iter_binary_vars():
        solution += str(int(np.round(docplex_sol.get_value(ii), 1)))
    return solution, docplex_sol

def landscape_results(
    cases,
    qubo_cases,
    classical_sol,
    betas=np.linspace(-np.pi / 2, 0, 30),
    gammas=np.linspace(-np.pi, 0, 50),
    normalized=-1,
    periodic=False,
):
    landscape = {}
    qaoa = {}
    for case in cases:
        print(f"case number: {case}")
        qaoa[case], setup = qaoa_model(
            qubo_cases[case], normalized=normalized, periodic=periodic
        )
        qaoa[case].optimize()
        landscape[case] = {}
        landscape[case]["results"] = qaoa[case].results
        landscape[case]["lowest"] = qaoa[case].results.lowest_cost_bitstrings(2**len(classical_sol[case]["sol"]))
        landscape[case]["energy"] = landscape_energy(qaoa[case], betas, gammas)
        landscape[case]["probability"] = landscape_probability(
            qaoa[case], betas, gammas, classical_sol[case]["sol"]
        )
        landscape[case]["optimal_position"] = (
            qaoa[case]
            .results.lowest_cost_bitstrings(100000)["solutions_bitstrings"]
            .index(classical_sol[case]["sol"])
        )
        landscape[case]["args"] = landscape_argmin(landscape[case])
        landscape[case]["setup"] = setup
        landscape[case]["CoP"] = landscape[case]["args"]["energy"]["probability"] * 2**len(classical_sol[case]["sol"])
        landscape[case]["Emin"] = np.min(landscape[case]["lowest"]["bitstrings_energies"])
        landscape[case]["Emax"] = np.max(landscape[case]["lowest"]["bitstrings_energies"])
        landscape[case]["Ef"] = np.min(landscape[case]["energy"])
        landscape[case]["r"] = (landscape[case]["Ef"] - landscape[case]["Emax"])/(landscape[case]["Emin"] - landscape[case]["Emax"])
    return landscape

def qaoa_results(penalty, mdl, sol_str, p=1, t=0.1, maxiter=100, unbalanced=False):
    ising = FromDocplex2IsingModel(
                mdl,
                multipliers=penalty[0],
                unbalanced_const=unbalanced,
                strength_ineq=[penalty[1],penalty[2]]
                ).ising_model

    qaoa = QAOA()
    qaoa.set_circuit_properties(p=p, init_type="ramp", linear_ramp_time=t)
    qaoa.set_classical_optimizer(maxiter=maxiter)
    qaoa.compile(ising)
    qaoa.optimize()
    results = qaoa.results.lowest_cost_bitstrings(2**mdl.number_of_binary_variables)
    results["openqaoa"] = qaoa.results
    results["pos"] = results["solutions_bitstrings"].index(sol_str)

    results["probability"] = results["probabilities"][results["pos"]]
    results["classical"] = sol_str
    results["CoP"] = results["probability"] * 2 ** len(sol_str)
    results["Emin"] = np.min(results["bitstrings_energies"])
    results["Emax"] = np.max(results["bitstrings_energies"])
    results["Ef"] = np.min(qaoa.results.optimized["optimized cost"])
    results["r"] = (results["Ef"] - results["Emax"])/(results["Emin"] - results["Emax"])
    results["opt_angles"] = qaoa.results.optimized["optimized angles"]
    return results


def normalizing(problem, normalized=-1, periodic=False):
    abs_weights = np.unique(np.abs(problem.weights))
    arg_sort = np.argsort(abs_weights)
    max_weight = abs_weights[arg_sort[normalized]]
    new_problem = copy(problem)
    if periodic:
        new_problem.weights = [weight // max_weight for weight in new_problem.weights]
    else:
        new_problem.weights = [weight / max_weight for weight in new_problem.weights]
    new_problem.constant /= max_weight
    return new_problem

def qaoa_model(problem, p=1, device=None, normalized=-1, periodic=False):
    if device == None:
        qaoa = QAOA()
    else:
        qaoa = QAOA(device)
    norm_problem = normalizing(problem, normalized, periodic=periodic)
    setup = {}
    setup["weights"] = norm_problem.weights
    setup["terms"] = norm_problem.terms
    qaoa.set_circuit_properties(p=p)
    qaoa.set_classical_optimizer(maxiter=1)
    qaoa.compile(norm_problem)
    return qaoa, setup

def landscape_energy(qaoa_mdl, betas, gammas):
    n_betas = len(betas)
    n_gammas = len(gammas)
    landscape = np.array(
        Parallel(n_jobs=num_cores)(
            delayed(cost_energy)(params, qaoa_mdl)
            for params in itertools.product(betas, gammas)
        )
    )
    # landscape =[ ]
    # for params in itertools.product(betas, gammas):
    #     landscape.append(cost_energy(params, qaoa_mdl))
    landscape = np.array(landscape)
    landscape = landscape.reshape((n_betas, n_gammas))
    return landscape

def landscape_probability(qaoa_mdl, betas, gammas, optimal):
    n_betas = len(betas)
    n_gammas = len(gammas)
    landscape =[ ]

    for params in itertools.product(betas, gammas):
        landscape.append(optimal_prob(params, qaoa_mdl, optimal))
    landscape = np.array(landscape)
    
    # landscape = np.array(
    #     Parallel(n_jobs=num_cores)(
    #         delayed(optimal_prob)(params, qaoa_mdl, optimal)
    #         for params in itertools.product(betas, gammas)
    #     )
    # )
    landscape = landscape.reshape((n_betas, n_gammas))
    return landscape

def landscape_argmin(landscape):
    args = {}
    for name in ["energy", "probability"]:
        ly, lx = landscape[name].shape
        if name == "energy":
            name2 = "probability"
            ll = np.argmin(landscape[name])
        else:
            name2 = "energy"
            ll = np.argmax(landscape[name])
        land2 = landscape[name2]
        nX, nY = ll % lx, ll // lx
        second = land2[nY, nX]
        args[name] = {"yx": (nY, nX), name2: second, name: landscape[name][nY, nX]}
    return args

def optimal_prob(params, qaoa, optimal):
    varational_params = qaoa.variate_params
    varational_params.update_from_raw(params)
    probs_set = qaoa.backend.probability_dict(varational_params)[optimal]
    return probs_set

def cost_energy(params, qaoa):
    varational_params = qaoa.variate_params
    varational_params.update_from_raw(params)
    return qaoa.backend.expectation(varational_params)
# =============================================================================
# Utils plots - from Portfolio Optimization via Quantum Zeno Dynamics on a Quantum Processor
# =============================================================================
pretty_print_dict = {
    'ar_projected' : r'$r$',
    'ar' : r'$r_{\mbox{penalty}}$' if usetex else r'r_{penalty}',
    'p_success' : r'$\delta$',
    'eta' : r'$\eta$'
}

def get_sequence_from_df(df, key, num_assets, mixer, max_nt=5):
    df_tmp = df[(df['num_assets'] == num_assets) & (df['mixer'] == mixer) & (df['nt'] <= max_nt)]
    if len(df_tmp) != max_nt:
        warnings.warn(f'Not enough results for {(key, num_assets, mixer)}; found {len(df_tmp)} != {max_nt}')
    return df_tmp.sort_values('nt')[key].tolist()

def plot_results(ax, d_penalty, d_zeno, key_to_plot, num_assets, mixer_name, max_nt, color_idx):
    alpha=1
    marker = 'P' if mixer_name == 'plus' else 'X'
    ax.plot(
        get_sequence_from_df(d_penalty, key_to_plot, num_assets, mixer_name, max_nt=max_nt), 
        label=f'{num_assets} penalty', linestyle='dotted', c=palette[color_idx], marker=marker, alpha=alpha, lw=lw
    )
    ax.plot(
        get_sequence_from_df(d_zeno, key_to_plot, num_assets, mixer_name, max_nt=max_nt), 
        label=f'{num_assets} zeno', c=palette[color_idx], marker=marker, alpha=alpha, lw=lw
    )
# =============================================================================
# Portfolio Optimization
# =============================================================================
def mu_fun(data, holding_period):
    """
    assets’ forecast returns at time t

    Parameters
    ----------
    data : np.array(num_time_steps)
        Price of the asset.
    holding_period: period to divide the data.

    Returns
    -------
    None.

    """
    min_t = min([len(d) for d in data])
    num_assets = len(data)
    mu = []
    for asset in range(num_assets):
        mu.append([data[asset][t+1]/data[asset][t] - 1 if data[asset][t] != 0 else 1 for t in range(min_t-1)])
    mu = np.array(mu)
    split =  min_t // holding_period
    mus = np.array([mu[:,i * holding_period:(i+1) * holding_period].sum(axis=1) for i in range(split)])
    return np.array(mus)

def cov_matrix(data, holding_period):
    min_t = min([len(d) for d in data])
    num_assets = len(data)
    mu = []
    for asset in range(num_assets):
        mu.append([data[asset][t+1]/data[asset][t] - 1 if data[asset][t] != 0 else 1 for t in range(min_t-1)])
    mu = np.array(mu)
    split =  min_t // holding_period
    cov =  [np.cov(mu[:,i*holding_period:(i+1)*holding_period], rowvar=True) for i in range(split)]
    return np.array(cov)
