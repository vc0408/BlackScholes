import black_scholes as bs

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns




st.header("Black Scholes European Option Calculator")


with st.form("input"):

    ticker = st.text_input("Stock Ticker: ")
    stock_price = st.number_input("Stock Price: ")
    strike_price = st.number_input("Strike Price: ")
    time = st.number_input("Time to maturity (Years): ")
    interest = st.number_input("Risk free interest rate (%): ")

    callput = st.radio("Call or Put? ", ["Call", "Put"], horizontal= True)

    submitted = st.form_submit_button("Calculate Black-Scholes Price")

if submitted:
    try:
        conv_interest = interest / 100

        vol = bs.get_sigma(ticker)

        bs_price, delta, gamma, vega, theta, rho, implied_vol = bs.get_data(
            callput, stock_price, strike_price, time, conv_interest, vol
        )

        data = {
            "Metric": ["Black-Scholes Price", "Delta", "Gamma", "Vega", "Theta", "Rho", "Implied Vol"],
            "Value": [f"${bs_price:.2f}", f"{delta:.4f}", f"{gamma:.4f}", f"{vega:.4f}", f"{theta:.4f}", f"{rho:.4f}", f"{implied_vol*100:.2f}%"]
        }

        df = pd.DataFrame(data)
        st.subheader("Option Greeks & Price")
        st.dataframe(df)

        M = 5000  # number of simulations
        N = 252   # number of time steps

        S_PATH = bs.simulate_paths(stock_price, time, conv_interest, vol, M, N)
        option_price_mc, discounted_payoffs = bs.mc_option_price(S_PATH, strike_price, time, conv_interest, callput)

        st.subheader("Monte Carlo Option Price")
        st.metric(f"{callput} Option Price", f"${option_price_mc:.2f}")

        # Stock paths plot
        st.subheader("Monte Carlo Stock Price Paths")
        plt.figure(figsize=(10, 5))
        plt.plot(S_PATH.T, color="skyblue", alpha=0.1)
        plt.xlabel("Time Steps")
        plt.ylabel("Stock Price")
        st.pyplot(plt)

        # Terminal stock price distribution
        st.subheader("Distribution of Terminal Stock Prices")
        plt.figure(figsize=(10, 5))
        sns.histplot(S_PATH[:, -1], bins=50, kde=True, color="lightgreen")
        plt.xlabel("Terminal Stock Price")
        plt.ylabel("Frequency")
        st.pyplot(plt)

    except Exception as e:
        st.error(f"Error calculating option: {e}")


