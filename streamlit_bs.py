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
            "Value": [f"${bs_price:.2f}", f"{delta:.4f}", f"{gamma:.4f}", f"{vega / 100:.4f}", f"{theta:.4f}", f"{rho / 100:.4f}", f"{implied_vol*100:.2f}%"]
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

    
        ## Volatility Smile

        st.subheader("Volatility Smile")

        iv_func = bs.getImpliedVolCall if callput == "Call" else bs.getImpliedVolPut

        baseline_value = strike_price
        step_pct = 0.04
        num_steps = 5

        start = baseline_value * (1 - step_pct * num_steps)
        end   = baseline_value * (1 + step_pct * num_steps)

        x = np.linspace(start, end, 1000)

        def safe_iv(K):
            try:
                v = iv_func(bs_price, stock_price, float(K), conv_interest, time_exp)

                return np.nan if v is None or (isinstance(v, (float, int)) and not np.isfinite(v)) else v
            except Exception:
                return np.nan

        y = np.array([safe_iv(k) for k in x])

        step_points = np.array([baseline_value * (1 + step_pct * k) for k in range(-num_steps, num_steps + 1)])
        step_values = np.array([safe_iv(k) for k in step_points])

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, y, label="Implied Volatility (smile)")
        ax.scatter(step_points, step_values, s=35, label="4% step strikes")
        ax.axvline(baseline_value, linestyle="--", label="Baseline strike")
        ax.set_title(f"IV vs Strike around {baseline_value} (±{num_steps}×{int(step_pct*100)}%)")
        ax.set_xlabel("Strike (K)")
        ax.set_ylabel("Implied Volatility")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error calculating option: {e}")


