# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Power Forecasting", page_icon="📈", layout="wide")

# Initialize session state keys (if not already)
for key, val in {
    'df': None,
    'freq': None,
    'n_future': 3,
    'selected_state': None,
    'models_results': None,
    'forecast_df': None
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# -------------------- TABS --------------------
tab1, tab2, tab3 = st.tabs(["📂 Dataset", "📈 Forecasting", "📊 Comparison"])

# -------------------- TAB 1: FIXED DATA (NO UPLOAD) --------------------
with tab1:
    st.header("📂 Dataset Preview (Fixed Data Loaded Automatically)")
    st.write("This app uses the fixed dataset `For Forecasting(web).xlsx` placed next to this script.")
    try:
        df_raw = pd.read_excel("For Forecasting(web).xlsx", header=0)
        df_raw.columns = [str(col).strip() for col in df_raw.columns]
        df_raw = df_raw.dropna(axis=1, how='all')
        st.success("✅ Loaded fixed dataset: For Forecasting(web).xlsx")
        st.subheader("Data Preview (First 10 Rows)")
        st.dataframe(df_raw.head(10))
    except FileNotFoundError:
        st.error("❌ 'For Forecasting(web).xlsx' not found in same folder. Place it beside this script and rerun.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # Identify State/UT column
    state_col = None
    possible_state_cols = ['STATES / UTs', 'State', 'STATES/UTs', 'states', 'STATES_UTs', 'STATES / UTs ']
    for col in possible_state_cols:
        if col in df_raw.columns:
            state_col = col
            break

    col_options = df_raw.columns.tolist()

    if state_col:
        state_options = sorted(df_raw[state_col].dropna().unique())
        selected_state = st.selectbox("Select State/UT (to filter)", state_options, key='fixed_state_select')
        st.session_state.selected_state = selected_state
    else:
        selected_state = None
        st.info("No state column detected — data will be aggregated across rows by date.")

    # Date column detection
    date_options_candidates = [
        c for c in col_options
        if c.lower() in ['year', 'date', 'time'] or 'year' in c.lower() or 'date' in c.lower()
    ]
    if not date_options_candidates:
        date_options_candidates = col_options

    default_date_col = 'Year' if 'Year' in col_options else date_options_candidates[0]
    date_col = st.selectbox("Select Date Column", col_options, index=col_options.index(default_date_col), key='fixed_date_select')

    # Value column selection
    exclude_cols = [date_col] + ([state_col] if state_col else [])
    value_options = [col for col in col_options if col not in exclude_cols]
    if not value_options:
        st.error("No numeric columns available after selecting date column.")
        st.stop()
    default_value_col = 'Total' if 'Total' in value_options else value_options[0]
    value_col = st.selectbox("Select Value Column", value_options, index=value_options.index(default_value_col), key='fixed_value_select')

    # Process button
    if st.button("Process Data"):
        df = df_raw.copy()
        if selected_state is not None and state_col:
            df = df[df[state_col] == selected_state].copy()
            if len(df) == 0:
                st.error(f"No data found for '{selected_state}'.")
                st.stop()
            st.success(f"Filtered to **{selected_state}** — {len(df)} rows")

        # Clean and format
        df[date_col] = pd.to_datetime(df[date_col].astype(str).str.strip(), errors='coerce')
        if df[date_col].isna().all():
            # try extract year
            df[date_col] = pd.to_datetime(df[date_col].astype(str).str.extract(r'(\d{4})')[0] + '-01-01', errors='coerce')

        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna(subset=[date_col, value_col])
        df = df.groupby(date_col)[value_col].sum().reset_index()
        df = df.set_index(date_col).sort_index()
        df = df.rename(columns={value_col: 'Value'})
        df = df[~df.index.duplicated(keep='first')]

        # Detect frequency roughly
        diffs = df.index.to_series().diff().dt.days.dropna()
        freq = None
        n_future = 3
        if len(diffs) > 1:
            avg_diff = diffs.mean()
            if 300 < avg_diff < 400:
                freq = 'A'
                n_future = 3
                st.info("✅ Detected **yearly** data.")
            elif 25 < avg_diff <= 35:
                freq = 'M'
                n_future = 12
                st.info("✅ Detected **monthly** data.")
            else:
                freq = None
                n_future = max(2, int(len(df) * 0.1))
                st.warning("⚠️ Irregular intervals detected.")
        else:
            freq = None
            n_future = 3

        st.session_state.df = df
        st.session_state.freq = freq
        st.session_state.n_future = n_future
        st.session_state.models_results = None
        st.session_state.forecast_df = None

        st.success(f"✅ Data processed successfully! {len(df)} valid rows.")
        st.dataframe(df)

# -------------------- TAB 2: FORECASTING --------------------
with tab2:
    st.header("📈 Forecasting & Auto Best-Fit Model")
    if st.session_state.df is None:
        st.warning("⚠️ Please process data in the Dataset tab first (click 'Process Data').")
    else:
        df = st.session_state.df.copy()
        freq = st.session_state.freq
        n_future = st.session_state.n_future

        # Stationarity check
        st.subheader("Stationarity Check (ADF)")
        try:
            adf_result = adfuller(df['Value'].values)
            pval = adf_result[1]
            is_stationary = pval < 0.05
            st.metric("ADF p-value", f"{pval:.4f}", delta="Stationary" if is_stationary else "Non-stationary")
        except Exception:
            pval = np.nan
            is_stationary = False
            st.info("ADF test could not be computed on this series.")

        # Transform if needed
        transform = 'none'
        values = df['Value'].values.copy()
        if not is_stationary and np.all(values > 0):
            values_log = np.log(values)
            transform = 'log'
            st.info("Log transformation applied for modeling.")
        else:
            values_log = values

        st.markdown("---")
        st.subheader("Run Single Forecast (choose method)")

        method = st.selectbox("Select Method", ["Auto-Best (recommended)", "Linear", "Quadratic", "Exponential", "Holt", "ARIMA", "Simple Moving Average"])
        run_button = st.button("Run Forecast")

        # helper functions
        def calc_metrics(true, pred):
            true = np.asarray(true)
            pred = np.asarray(pred)
            if len(true) == 0 or len(pred) == 0:
                return np.nan, np.nan, np.nan
            rmse = np.sqrt(mean_squared_error(true, pred))
            nrmse = rmse / np.mean(true) if np.mean(true) != 0 else np.nan
            mape = mean_absolute_percentage_error(true, pred) * 100 if np.all(true != 0) else np.nan
            return rmse, nrmse, mape

        # prepare train/test split
        n_rows = len(df)
        split = max(3, int(n_rows * 0.7))
        train_vals = values_log[:split]
        test_vals = values[split:]  # keep test in original scale
        x_train = np.arange(len(train_vals))
        x_test = np.arange(len(train_vals), len(values_log))

        models_results = {}

        if run_button:
            # 1) Linear
            try:
                coef_lin = np.polyfit(x_train, train_vals, 1)
                pred_test_lin_log = np.polyval(coef_lin, x_test) if len(x_test) > 0 else np.array([])
                pred_test_lin = np.exp(pred_test_lin_log) if transform == 'log' else pred_test_lin_log
                rmse_lin, nrmse_lin, mape_lin = calc_metrics(test_vals, pred_test_lin) if len(test_vals) > 0 else (np.nan, np.nan, np.nan)
                models_results['Linear'] = {'rmse': rmse_lin, 'nrmse': nrmse_lin, 'mape': mape_lin, 'fitted': coef_lin}
            except Exception:
                models_results['Linear'] = {'rmse': np.nan, 'nrmse': np.nan, 'mape': np.nan, 'fitted': None}

            # 2) Quadratic
            try:
                if len(x_train) >= 3:
                    coef_quad = np.polyfit(x_train, train_vals, 2)
                    pred_test_quad_log = np.polyval(coef_quad, x_test) if len(x_test) > 0 else np.array([])
                    pred_test_quad = np.exp(pred_test_quad_log) if transform == 'log' else pred_test_quad_log
                    rmse_q, nrmse_q, mape_q = calc_metrics(test_vals, pred_test_quad) if len(test_vals) > 0 else (np.nan, np.nan, np.nan)
                    models_results['Quadratic'] = {'rmse': rmse_q, 'nrmse': nrmse_q, 'mape': mape_q, 'fitted': coef_quad}
                else:
                    models_results['Quadratic'] = {'rmse': np.nan, 'nrmse': np.nan, 'mape': np.nan, 'fitted': None}
            except Exception:
                models_results['Quadratic'] = {'rmse': np.nan, 'nrmse': np.nan, 'mape': np.nan, 'fitted': None}

            # 3) Exponential (fit on original if possible)
            try:
                if np.all(values > 0):
                    coef_exp = np.polyfit(x_train, np.log(values[:split]), 1)
                    pred_test_exp_log = np.polyval(coef_exp, x_test) if len(x_test) > 0 else np.array([])
                    pred_test_exp = np.exp(pred_test_exp_log)
                    rmse_e, nrmse_e, mape_e = calc_metrics(test_vals, pred_test_exp) if len(test_vals) > 0 else (np.nan, np.nan, np.nan)
                    models_results['Exponential'] = {'rmse': rmse_e, 'nrmse': nrmse_e, 'mape': mape_e, 'fitted': coef_exp}
                else:
                    models_results['Exponential'] = {'rmse': np.nan, 'nrmse': np.nan, 'mape': np.nan, 'fitted': None}
            except Exception:
                models_results['Exponential'] = {'rmse': np.nan, 'nrmse': np.nan, 'mape': np.nan, 'fitted': None}

            # 4) Holt
            try:
                if len(train_vals) >= 5:
                    train_series = pd.Series(train_vals, index=df.index[:split])
                    holt_model = ExponentialSmoothing(train_series, trend='add', seasonal=None, initialization_method='estimated').fit()
                    pred_test_holt_log = holt_model.forecast(steps=max(0, len(test_vals)))
                    pred_test_holt = np.exp(pred_test_holt_log) if transform == 'log' else pred_test_holt_log
                    rmse_h, nrmse_h, mape_h = calc_metrics(test_vals, pred_test_holt) if len(test_vals) > 0 else (np.nan, np.nan, np.nan)
                    models_results['Holt'] = {'rmse': rmse_h, 'nrmse': nrmse_h, 'mape': mape_h, 'fitted': holt_model}
                else:
                    models_results['Holt'] = {'rmse': np.nan, 'nrmse': np.nan, 'mape': np.nan, 'fitted': None}
            except Exception:
                models_results['Holt'] = {'rmse': np.nan, 'nrmse': np.nan, 'mape': np.nan, 'fitted': None}

            # 5) ARIMA
            try:
                if len(train_vals) >= 5:
                    # ARIMA expects 1d array (we pass original or transformed series depending)
                    arima_train = pd.Series(train_vals, index=df.index[:split])
                    arima_model = ARIMA(arima_train, order=(1, 1, 1)).fit()
                    pred_test_arima_log = arima_model.forecast(steps=max(0, len(test_vals)))
                    pred_test_arima = np.exp(pred_test_arima_log) if transform == 'log' else pred_test_arima_log
                    rmse_a, nrmse_a, mape_a = calc_metrics(test_vals, pred_test_arima) if len(test_vals) > 0 else (np.nan, np.nan, np.nan)
                    models_results['ARIMA'] = {'rmse': rmse_a, 'nrmse': nrmse_a, 'mape': mape_a, 'fitted': arima_model}
                else:
                    models_results['ARIMA'] = {'rmse': np.nan, 'nrmse': np.nan, 'mape': np.nan, 'fitted': None}
            except Exception:
                models_results['ARIMA'] = {'rmse': np.nan, 'nrmse': np.nan, 'mape': np.nan, 'fitted': None}

            # 6) Simple Moving Average (SMA)
            try:
                window = st.slider("SMA window (for Simple Moving Average)", 2, min(12, max(2, len(df)//2)), 3, key='sma_window')
                sma_series = pd.Series(df['Value']).rolling(window=window).mean()
                sma_last = sma_series.iloc[-1]
                sma_forecast_vals = np.array([sma_last] * len(test_vals)) if len(test_vals) > 0 else np.array([])
                rmse_s, nrmse_s, mape_s = calc_metrics(test_vals, sma_forecast_vals) if len(test_vals) > 0 else (np.nan, np.nan, np.nan)
                models_results['SMA'] = {'rmse': rmse_s, 'nrmse': nrmse_s, 'mape': mape_s, 'fitted': {'window': window, 'last': sma_last}}
            except Exception:
                models_results['SMA'] = {'rmse': np.nan, 'nrmse': np.nan, 'mape': np.nan, 'fitted': None}

            st.session_state.models_results = models_results

            # If Auto-Best selected, pick best and generate forecast
            if method == "Auto-Best (recommended)":
                valid_rmse = {k: v['rmse'] for k, v in models_results.items() if np.isfinite(v['rmse'])}
                if valid_rmse:
                    best_model = min(valid_rmse, key=lambda k: valid_rmse[k])
                    st.success(f"🏆 Best model by RMSE: **{best_model}**  (RMSE = {valid_rmse[best_model]:.3f})")
                    selected_forecast_model = best_model
                else:
                    st.warning("No valid RMSE values to choose best model.")
                    selected_forecast_model = None
            else:
                selected_forecast_model = method

            # Forecast next n_future using selected_forecast_model
            if selected_forecast_model:
                # Build future index
                last_date = df.index[-1]
                if freq == 'A':
                    future_index = pd.date_range(start=last_date + pd.offsets.YearEnd(1), periods=n_future, freq='A')
                elif freq == 'M':
                    future_index = pd.date_range(start=last_date + pd.offsets.MonthEnd(1), periods=n_future, freq='M')
                else:
                    # fallback yearly
                    future_index = pd.date_range(start=last_date + pd.DateOffset(years=1), periods=n_future, freq='Y')

                x_future = np.arange(len(values), len(values) + n_future)

                future_pred = np.full(n_future, np.nan)

                try:
                    if selected_forecast_model == 'Linear' and models_results['Linear']['fitted'] is not None:
                        future_pred_log = np.polyval(models_results['Linear']['fitted'], x_future)
                        future_pred = np.exp(future_pred_log) if transform == 'log' else future_pred_log
                    elif selected_forecast_model == 'Quadratic' and models_results['Quadratic']['fitted'] is not None:
                        future_pred_log = np.polyval(models_results['Quadratic']['fitted'], x_future)
                        future_pred = np.exp(future_pred_log) if transform == 'log' else future_pred_log
                    elif selected_forecast_model == 'Exponential' and models_results['Exponential']['fitted'] is not None:
                        future_pred_log = np.polyval(models_results['Exponential']['fitted'], x_future)
                        future_pred = np.exp(future_pred_log)
                    elif selected_forecast_model == 'Holt' and models_results['Holt']['fitted'] is not None:
                        holt_full = ExponentialSmoothing(values_log if transform == 'log' else values, trend='add', seasonal=None, initialization_method='estimated').fit()
                        fut = holt_full.forecast(steps=n_future)
                        future_pred = np.exp(fut) if transform == 'log' else fut
                    elif selected_forecast_model == 'ARIMA' and models_results['ARIMA']['fitted'] is not None:
                        arima_full = ARIMA(values_log if transform == 'log' else values, order=(1,1,1)).fit()
                        fut = arima_full.forecast(steps=n_future)
                        future_pred = np.exp(fut) if transform == 'log' else fut
                    elif selected_forecast_model == 'SMA' and models_results['SMA']['fitted'] is not None:
                        future_pred = np.array([models_results['SMA']['fitted']['last']] * n_future)
                    else:
                        # if not available, fallback to last observed
                        future_pred = np.array([values[-1]] * n_future)
                except Exception:
                    future_pred = np.array([values[-1]] * n_future)

                forecast_df = pd.DataFrame({'Forecast': future_pred}, index=future_index)
                st.session_state.forecast_df = forecast_df

                # plot actual + forecast
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df['Value'], name='Actual', mode='lines+markers'))
                fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], name=f'{selected_forecast_model} Forecast', mode='lines+markers'))
                fig.update_layout(title=f"Forecast ({selected_forecast_model})", xaxis_title="Date", yaxis_title="Value", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Forecast Table")
                out_df = forecast_df.reset_index().rename(columns={'index': 'Date', 'Forecast': 'Forecast Value'})
                st.dataframe(out_df.style.format({'Forecast Value': '{:.2f}'}))

        # If models_results exist from previous runs, show RMSE comparison
        if st.session_state.models_results:
            st.markdown("---")
            st.subheader("📊 RMSE Comparison (latest run)")
            res = st.session_state.models_results
            rmse_df = pd.DataFrame([
                {'Model': k, 'RMSE': (v['rmse'] if v else np.nan), 'NRMSE': (v.get('nrmse', np.nan) if isinstance(v, dict) else np.nan),
                 'MAPE (%)': (v.get('mape', np.nan) if isinstance(v, dict) else np.nan)}
                for k, v in res.items()
            ])
            rmse_df = rmse_df.sort_values('RMSE', na_position='last').reset_index(drop=True)
            st.dataframe(rmse_df.style.highlight_min(subset=['RMSE'], color='#d4edda'))

# -------------------- TAB 3: COMPARISON --------------------
with tab3:
    st.header("📊 Model Comparison & Multi-Model Forecasts")
    if st.session_state.df is None:
        st.warning("⚠️ Please process data in the Dataset tab first.")
    else:
        df = st.session_state.df.copy()
        freq = st.session_state.freq
        n_future = st.session_state.n_future

        if st.session_state.models_results is None:
            st.info("Run forecasting in the Forecasting tab first (click 'Run Forecast').")
        else:
            res = st.session_state.models_results

            # Build forecasts for each model (if fitted present) for next n_future
            last_date = df.index[-1]
            if freq == 'A':
                future_index = pd.date_range(start=last_date + pd.offsets.YearEnd(1), periods=n_future, freq='A')
            elif freq == 'M':
                future_index = pd.date_range(start=last_date + pd.offsets.MonthEnd(1), periods=n_future, freq='M')
            else:
                future_index = pd.date_range(start=last_date + pd.DateOffset(years=1), periods=n_future, freq='Y')

            forecasts = {}
            for name, info in res.items():
                try:
                    if name == 'Linear' and info['fitted'] is not None:
                        coef = info['fitted']
                        x_future = np.arange(len(df), len(df) + n_future)
                        fut = np.polyval(coef, x_future)
                        # if original transform used log, we already used test/fit on transformed series earlier; but here we return on original scale
                        if np.all(df['Value'] > 0) and np.nanmean(np.log(df['Value'])) == np.nanmean(np.log(df['Value'])):
                            # assume exponential/log used earlier - cannot be certain; safe to use original linear (best-effort)
                            pass
                        forecasts[name] = fut
                    elif name == 'Quadratic' and info['fitted'] is not None:
                        coef = info['fitted']
                        x_future = np.arange(len(df), len(df) + n_future)
                        forecasts[name] = np.polyval(coef, x_future)
                    elif name == 'Exponential' and info['fitted'] is not None:
                        coef = info['fitted']
                        x_future = np.arange(len(df), len(df) + n_future)
                        forecasts[name] = np.exp(np.polyval(coef, x_future))
                    elif name == 'Holt' and info['fitted'] is not None:
                        holt_full = ExponentialSmoothing(df['Value'], trend='add', seasonal=None, initialization_method='estimated').fit()
                        forecasts[name] = holt_full.forecast(steps=n_future)
                    elif name == 'ARIMA' and info['fitted'] is not None:
                        arima_full = ARIMA(df['Value'], order=(1,1,1)).fit()
                        forecasts[name] = arima_full.forecast(steps=n_future)
                    elif name == 'SMA' and info['fitted'] is not None:
                        forecasts[name] = np.array([info['fitted']['last']] * n_future)
                    else:
                        forecasts[name] = np.array([np.nan]*n_future)
                except Exception:
                    forecasts[name] = np.array([np.nan]*n_future)

            # Compose comparison dataframe
            comp_df = pd.DataFrame(index=future_index)
            for k, v in forecasts.items():
                comp_df[k] = v

            # Make comparison table with RMSE and eq (if available)
            metrics_rows = []
            for k, v in res.items():
                rm = v.get('rmse', np.nan) if isinstance(v, dict) else np.nan
                nr = v.get('nrmse', np.nan) if isinstance(v, dict) else np.nan
                mp = v.get('mape', np.nan) if isinstance(v, dict) else np.nan
                eq = ''
                fitted = v.get('fitted', None) if isinstance(v, dict) else None
                if isinstance(fitted, (list, np.ndarray, np.generic)):
                    eq = 'Coef: ' + np.array2string(np.asarray(fitted), precision=4, max_line_width=200)
                elif hasattr(fitted, '__class__') and fitted is not None:
                    eq = str(type(fitted)).split("'")[1]
                elif isinstance(fitted, dict):
                    eq = str(fitted)
                metrics_rows.append({'Model': k, 'eq': eq, 'RMSE': rm, 'NRMSE': nr, 'MAPE (%)': mp})

            metrics_df = pd.DataFrame(metrics_rows).sort_values('RMSE', na_position='last').reset_index(drop=True)

            st.subheader("Model Metrics")
            st.dataframe(metrics_df.style.highlight_min(subset=['RMSE'], color='#d4edda'))

            st.subheader(f"Forecasts (next {n_future} periods)")
            # Format numeric and show
            st.dataframe(comp_df.reset_index().rename(columns={'index': 'Date'}).style.format({c: '{:.2f}' for c in comp_df.columns}))

            # Chart
            st.subheader("Forecast Comparison Chart")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Value'], name='Actual', mode='lines', line=dict(width=3)))
            for col in comp_df.columns:
                fig.add_trace(go.Scatter(x=comp_df.index, y=comp_df[col], name=col, mode='lines+markers'))
            fig.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Value")
            st.plotly_chart(fig, use_container_width=True)


