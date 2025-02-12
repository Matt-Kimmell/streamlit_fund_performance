import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Set Streamlit page configuration
st.set_page_config(page_title="Fund Performance Analysis", layout="wide")

# File Upload
st.title("Fund Performance Analysis")
uploaded_file = st.file_uploader("Upload an Excel file with ticker data", type=["xlsx"])

if uploaded_file:
    # Load data
    df = pd.read_excel(uploaded_file, index_col=0)
    df.index = pd.to_datetime(df.index)

    # Align dates by keeping only common dates across all tickers
    df = df.dropna(how='any')

    # Sidebar options
    st.sidebar.header("Options")
    start_date = st.sidebar.date_input("Start Date", value=df.index.min())
    end_date = st.sidebar.date_input("End Date", value=df.index.max())
    risk_free_rate_annual = st.sidebar.number_input("Annual Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=4.0)
    risk_free_rate_daily = (risk_free_rate_annual / 100) / 252  # Convert to daily rate
    available_tickers = df.columns.tolist()
    selected_tickers = st.sidebar.multiselect("Select tickers to analyze", available_tickers, default=available_tickers)
    benchmark = st.sidebar.selectbox("Select a benchmark ticker", available_tickers)
    
    if start_date >= end_date:
        st.error("End date must be after start date.")
    else:
        
        if selected_tickers:
        # Filter data based on selected date range
            filtered_df = df[selected_tickers].loc[start_date:end_date]

        # Calculate daily returns
            daily_returns = filtered_df.pct_change().dropna()

        # Plot cumulative returns with Plotly
            cumulative_returns = (1 + daily_returns).cumprod()

            # Calculate monthly returns for each ticker and the benchmark
            monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        else:
            st.warning("Please select at least one ticker to proceed.")
        
        # Calculate cumulative return over the selected date range
        cumulative_return = (filtered_df.iloc[-1] / filtered_df.iloc[0] - 1).round(4) * 100

        st.write("---")

        # Create columns for ticker stats
        cols = st.columns(len(filtered_df.columns))  # One column per ticker

        for i, ticker in enumerate(filtered_df):
            with cols[i]:
                # Display cumulative return
                st.metric(label=f"{ticker} - Cumulative Return", value=f"{cumulative_return[ticker]:.2f}%")

        fig = go.Figure()
        for ticker in cumulative_returns.columns:
            fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns[ticker], mode='lines', name=ticker))
        
        fig.update_layout(
            title="Cumulative Returns Over Time",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            hovermode="x unified",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        def calculate_max_drawdown(cum_returns):
            """ Calculate max drawdown for a single ticker's cumulative returns. """
            drawdown = cum_returns / cum_returns.cummax() - 1
            return drawdown.min()


        def calculate_metrics(returns):
            excess_returns = returns - risk_free_rate_daily
            one_month = returns[-21:].sum() if len(returns) >= 21 else np.nan
            three_month = returns[-63:].sum() if len(returns) >= 63 else np.nan
            twelve_month = returns[-252:].sum() if len(returns) >= 252 else np.nan
            ytd = returns[returns.index.year == pd.Timestamp.now().year].sum()
            max_drawdown = calculate_max_drawdown(cumulative_returns[ticker])
            annualized_volatility = returns.std() * np.sqrt(252)
            
            # Sharpe Ratio over the full available range
            sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else np.nan
            
            # Sharpe Ratio over the last 12 months (last 252 trading days)
            last_12m_returns = returns[-252:] if len(returns) >= 252 else returns
            last_12m_excess_returns = last_12m_returns - risk_free_rate_daily
            sharpe_ratio_12m = (last_12m_excess_returns.mean() / last_12m_excess_returns.std()) * np.sqrt(252) if last_12m_excess_returns.std() > 0 else np.nan
            
            return one_month, three_month, twelve_month, ytd, max_drawdown, annualized_volatility, sharpe_ratio, sharpe_ratio_12m


        metrics = {
            "1M Return": [],
            "3M Return": [],
            "12M Return": [],
            "YTD Return": [],
            "Max Drawdown": [],
            "Annualized Volatility": [],
            "Sharpe Ratio over Range": [],
            "12M Sharpe Ratio": [],  # Add this new metric
        }

        for ticker in daily_returns.columns:
            one_month, three_month, twelve_month, ytd, max_drawdown, annualized_volatility, sharpe_ratio, sharpe_ratio_12m = calculate_metrics(daily_returns[ticker])
            metrics["1M Return"].append(one_month)
            metrics["3M Return"].append(three_month)
            metrics["12M Return"].append(twelve_month)
            metrics["YTD Return"].append(ytd)
            metrics["Max Drawdown"].append(max_drawdown)
            metrics["Annualized Volatility"].append(annualized_volatility)
            metrics["Sharpe Ratio over Range"].append(sharpe_ratio)
            metrics["12M Sharpe Ratio"].append(sharpe_ratio_12m)  # Append the new Sharpe Ratio



        # Display metrics as a table
        metrics_df = pd.DataFrame(metrics, index=daily_returns.columns)
        st.subheader("Performance Summary")
        st.dataframe(metrics_df.style.format({
            "1M Return": "{:.2%}",
            "3M Return": "{:.2%}",
            "12M Return": "{:.2%}",
            "YTD Return": "{:.2%}",
            "Max Drawdown": "{:.2%}",
            "Annualized Volatility": "{:.2%}",
            "Sharpe Ratio over Range": "{:.2f}",
            "12M Sharpe Ratio": "{:.2f}"
        }))
        # Remove the benchmark ticker from the selected tickers list (if present)
        filtered_tickers = [ticker for ticker in selected_tickers if ticker != benchmark]
        
        # Create a comparison dataframe showing outperformance/underperformance
        benchmark_returns = monthly_returns[benchmark]
        relative_performance = monthly_returns[filtered_tickers].sub(benchmark_returns, axis=0)
        outperformance = relative_performance > 0  # True if the ticker outperformed the benchmark

        # Format the results for display
        performance_table = relative_performance.applymap(lambda x: "Outperform" if x > 0 else "Underperform")
        num_months = performance_table.shape[0]
        outperformance_percentage = (performance_table == "Outperform").sum(axis=0) / num_months * 100
        
        st.subheader("Monthly Performance Relative to Benchmark")
        st.write("% of Months Outperforming")        
        cols = st.columns(len(filtered_tickers))  # One column per ticker

        for i, ticker in enumerate(filtered_tickers):
            with cols[i]:
                st.metric(label=f"{ticker}:", value=f"{outperformance_percentage[ticker]:.2f}%")

        
        
        
        def highlight_performance(val):
            if val == "Outperform":
                return "background-color: lightgreen"
            elif val == "Underperform":
                return "background-color: lightcoral"
            return ""
        


        fig = go.Figure()

        # Plot benchmark monthly returns
        fig.add_trace(go.Bar(
            x=monthly_returns.index,
            y=monthly_returns[benchmark],
            name=f"{benchmark} (Benchmark)",
            marker_color='blue'
        ))

        # Plot monthly returns for each selected ticker
        for ticker in filtered_tickers:
            fig.add_trace(go.Scatter(
                x=monthly_returns.index,
                y=monthly_returns[ticker],
                mode='lines+markers',
                name=ticker
            ))

        fig.update_layout(
            title="Monthly Returns Comparison with Benchmark",
            xaxis_title="Month",
            yaxis_title="Monthly Return",
            hovermode="x unified",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.write("Monthly Performance Details")
        st.dataframe(performance_table.style.applymap(highlight_performance))
