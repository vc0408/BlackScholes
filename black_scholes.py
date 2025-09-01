import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
import scipy.stats as sp
import yfinance as yf
import streamlit as st



def get_sigma(symbol): 
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period='1d')
    if todays_data.empty:
        raise ValueError(f"No data for ticker {symbol}")
    S = todays_data['Close'][0]
    
    today = datetime.today()
    last_year = today - relativedelta(years=1)
    
    data = yf.download(symbol, start=last_year, end=today)
    
    if data.empty:
        raise ValueError(f"No historical data for ticker {symbol}")
    
    if 'Adj Close' not in data.columns:
        if 'Close' in data.columns:
            data['Adj Close'] = data['Close']
        else:
            raise ValueError(f"No 'Adj Close' or 'Close' column for {symbol}")
    
    data['Daily Return'] = data['Adj Close'].pct_change()
    daily_vol = data['Daily Return'].std()
    sigma = daily_vol * (252 ** 0.5) 
    
    return sigma

def blackScholesCall(S, K, T, r, sigma):

  d1 = (np.log(S/K) + (r + 0.5 * sigma **2) * T) / (sigma * np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)

  return S * sp.norm.cdf(d1) - K * np.exp(-r * T) * sp.norm.cdf(d2)
    
def blackScholesPut(S, K, T, r, sigma):

  d1 = (np.log(S/K) + (r + 0.5 * sigma **2) * T) / (sigma * np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)

  return  K * np.exp(-r * T) * sp.norm.cdf(-d2) - S * sp.norm.cdf(-d1)

def delta_call(S, K, T, r, sigma):
  d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
  delta = sp.norm.cdf(d1)
  return delta

def delta_put(S, K, T, r, sigma):
  d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
  delta = -sp.norm.cdf(-d1)
  return delta

def gamma_(S, K, T, r, sigma):
  d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
  gamma = sp.norm.pdf(d1) / (S * sigma * np.sqrt(T))
  return gamma

def vega_(S, K, T, r, sigma):
  d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
  vega = S * sp.norm.pdf(d1) * np.sqrt(T)
  return vega


def theta_call(S, K, T, r, sigma):
  d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)

  theta_call = (-S * sp.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * sp.norm.cdf(d2))

  return theta_call / 365

def theta_put(S, K, T, r, sigma):
  d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)

  theta_put = (-S * sp.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * sp.norm.cdf(-d2))

  return theta_put / 365

def rho_call(S, K, T, r, sigma):
  d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)
  rho_call = K * T * np.exp(-r * T) * sp.norm.cdf(d2)
  return rho_call

def rho_put(S, K, T, r, sigma):
  d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)
  rho_put = -K * T * np.exp(-r * T) * sp.norm.cdf(-d2)
  return rho_put

def get_data(callput , S, K, T, r, sigma):

  if callput == "Call": 
    price = blackScholesCall(S, K, T, r, sigma)
    delta = delta_call(S, K, T, r, sigma) 
    gamma = gamma_(S, K, T, r, sigma)
    vega = vega_(S, K, T, r, sigma)
    theta = theta_call(S, K, T, r, sigma)
    rho = rho_call(S, K, T, r, sigma)
    implied_vol = getImpliedVolCall(price, S, K, r, T)

  else: 
    price = blackScholesPut(S, K, T, r, sigma)
    delta = delta_put(S, K, T, r, sigma) 
    gamma = gamma_(S, K, T, r, sigma)
    vega = vega_(S, K, T, r, sigma)
    theta = theta_put(S, K, T, r, sigma)
    rho = rho_call(S, K, T, r, sigma)
    implied_vol = getImpliedVolPut(price, S, K, r, T)

  return price, delta, gamma, vega, theta, rho, implied_vol

# Implied Volatility with Newton Raphson

def InflectionPoint(S, K, r, T):
  m = S / (K * np.exp(-r * T))
  return np.sqrt(2 * np.abs(np.log(m)) / T)

def getImpliedVolCall(C, S, K, r, T, epsilon = 1e-6):
  x0 = InflectionPoint(S, K, r, T)
  p = blackScholesCall(S, K, T, r, x0)
  v = vega_(S, K, T, r, x0)
  

  while (np.abs((p - C) / v) > epsilon):
    x0 = x0 - (p- C) / v
    p = blackScholesCall(S, K, T, r, x0)
    v = vega_(S, K, T, r, x0)

  return x0
   
def getImpliedVolPut(C, S, K, r, T, epsilon = 1e-6):

  x0 = InflectionPoint(S, K, r, T)
  p = blackScholesPut(S, K, T, r, x0)
  v = vega_(S, x0, K, r, T)

  while (np.abs((p - C) / v) > epsilon):
    x0 = x0 - (p- C) / v
    p = blackScholesPut(S, K, T, r, x0)
    v = vega_(S, K, T, r, x0)

  return x0


## Monte Carlo 

def simulate_paths(S0, T, r, sigma, M, N):
    dt = T / N
    nudt = (r - 0.5 * sigma**2) * dt
    sidt = sigma * np.sqrt(dt)
    
    Z = np.random.normal(size=(M, N))
    S_PATH = np.zeros((M, N+1), dtype=float)
    S_PATH[:, 0] = S0
    
    S_PATH[:, 1:] = S0 * np.exp(np.cumsum(nudt + sidt * Z, axis=1))
    return S_PATH

def mc_option_price(S_PATH, K, T, r, callput):
    if callput == "Call":
        payoffs = np.maximum(S_PATH[:, -1] - K, 0)
    else:
        payoffs = np.maximum(K - S_PATH[:, -1], 0)
    
    discounted_payoffs = np.exp(-r*T) * payoffs
    price = np.mean(discounted_payoffs)
    return price, discounted_payoffs


