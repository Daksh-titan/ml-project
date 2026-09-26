import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mqtf_core import (
    TREATMENTS,
    DAYS,
    FEATURES,
    DOMAIN_A,
    DOMAIN_B,
    CSVValidationError,
    default_arrays,
    default_long_dataframe,
    run_mqtf,
    monte_carlo_selection,
    validate_and_build,
    backtest,
    metrics_by_variable,
    feature_to_displacement,
)

st.set_page_config(page_title="MQTF Research Analysis Tool", page_icon="📊", layout="wide")

st.title("MQTF Research Analysis Tool")
st.caption("Multivariate Quality Trajectory Framework for fermented millet beverage data")

st.sidebar.header("Input")
use_example = st.sidebar.checkbox("Use verified built-in dataset", value=True)
upload = st.sidebar.file_uploader("Upload MQTF CSV", type=["csv"])

st.sidebar.header("Monte Carlo stability")
n_sim = st.sidebar.select_slider("Simulations", options=[100, 500, 1000, 2000], value=1000)
seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

st.sidebar.header("Validation")
run_forecast_validation = st.sidebar.checkbox("Run forecasting extension validation", value=False)

if upload is not None:
    try:
        df_input = pd.read_csv(upload)
        X_mean, X_sd, have_sd, warnings = validate_and_build(df_input)
        source_label = f"Uploaded CSV: {upload.name}"
    except CSVValidationError as e:
        st.error("CSV validation failed.")
        for issue in e.issues:
            st.write(f"- {issue}")
        st.stop()
    except Exception as e:
        st.error(f"Could not read the CSV: {e}")
        st.stop()
elif use_example:
    X_mean, X_sd = default_arrays()
    df_input = default_long_dataframe()
    have_sd = True
    warnings = []
    source_label = "Verified built-in dataset"
else:
    st.info("Upload a CSV or enable the verified built-in dataset.")
    st.stop()

for warning in warnings:
    st.warning(warning)

try:
    A, B, STD, FTD = run_mqtf(X_mean)
except Exception as e:
    st.error(f"MQTF calculation failed: {e}")
    st.stop()

st.success(f"Loaded {source_label}. Dataset shape: {len(TREATMENTS)} treatments × {len(DAYS)} days × {len(FEATURES)} variables.")

if have_sd:
    st.info("All 18 SD columns are available. Monte Carlo feature-selection stability is enabled.")
else:
    st.warning("No SD columns were supplied. Monte Carlo feature-selection stability is unavailable for this input and is not being replaced with zero uncertainty.")

if st.button("Run MQTF analysis", type="primary"):
    st.session_state["run_mc"] = True

run_mc = st.session_state.get("run_mc", True)

if run_mc and have_sd:
    with st.spinner(f"Running {n_sim:,} Monte Carlo feature-selection simulations..."):
        counts_A, counts_B = monte_carlo_selection(X_mean, X_sd, n_sim=n_sim, seed=int(seed))
else:
    counts_A, counts_B = None, None

regression_expected_std = np.array([4.352, 4.569, 4.134, 3.916, 4.134, 4.787])
regression_expected_ftd = np.array([0.021, 3.004, 3.047, 1.553, 3.015, 1.576])
regression_pass = np.allclose(STD[:, 4], regression_expected_std, atol=0.001) and np.allclose(FTD[:, 4], regression_expected_ftd, atol=0.001)

if regression_pass:
    st.success("MQTF reproducibility check: PASS — day-28 STD/FTD match the validated notebook within ±0.001.")
else:
    st.error("MQTF reproducibility check: FAIL — day-28 STD/FTD do not match the validated notebook.")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Trajectories", "MQTF Results", "Stability", "Validation"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Treatments", len(TREATMENTS))
    c2.metric("Storage days", len(DAYS))
    c3.metric("Variables", len(FEATURES))
    c4.metric("Replicates", 3)

    st.subheader("Selected features")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Domain A — physicochemical trajectory**")
        st.write(", ".join(A["best_names"]))
    with col2:
        st.write("**Domain B — functional trajectory**")
        st.write(", ".join(B["best_names"]))

    st.markdown("### What STD and FTD mean")
    st.write("STD is the multivariate displacement from the Day-0 baseline using the selected Domain A features. FTD is the corresponding displacement using the selected Domain B features.")
    st.write("They are trajectory-displacement measures within the observed 28-day experiment; they are not direct shelf-life predictions.")

with tab2:
    st.subheader("Selected feature trajectories")
    selected_features = list(dict.fromkeys(A["best_names"] + B["best_names"]))
    selected = st.selectbox("Variable", selected_features)
    j = FEATURES.index(selected)
    fig = go.Figure()
    for i, treatment in enumerate(TREATMENTS):
        fig.add_trace(go.Scatter(x=DAYS, y=X_mean[i, :, j], mode="lines+markers", name=treatment))
    fig.update_layout(xaxis_title="Storage day", yaxis_title=selected, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    metric_choice = st.selectbox("Trajectory displacement", ["STD", "FTD"])
    arr = STD if metric_choice == "STD" else FTD
    fig2 = go.Figure()
    for i, treatment in enumerate(TREATMENTS):
        fig2.add_trace(go.Scatter(x=DAYS, y=arr[i], mode="lines+markers", name=treatment))
    fig2.update_layout(xaxis_title="Storage day", yaxis_title=metric_choice, hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Day-28 MQTF results")
    results = pd.DataFrame({
        "Treatment": TREATMENTS,
        "STD": STD[:, 4],
        "FTD": FTD[:, 4],
    })
    st.dataframe(results, use_container_width=True, hide_index=True)

    st.download_button(
        "Download day-28 results CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="MQTF_day28_results.csv",
        mime="text/csv",
    )

    st.subheader("Feature-selection details")
    a_signal, a_redundancy = A["metrics"][A["best"]]
    b_signal, b_redundancy = B["metrics"][B["best"]]
    details = pd.DataFrame({
        "Domain": ["A", "B"],
        "Selected features": [" + ".join(A["best_names"]), " + ".join(B["best_names"])],
        "Signal": [a_signal, b_signal],
        "Redundancy": [a_redundancy, b_redundancy],
    })
    st.dataframe(details, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Monte Carlo feature-selection stability")
    st.write("This analysis perturbs each reported mean using its experimental standard error (SD / √3), reruns the complete feature-selection procedure, and counts how often each selected subset is reproduced.")
    st.warning("This is a feature-selection robustness check, not prediction accuracy and not a prediction confidence interval.")

    if not have_sd:
        st.info("Upload all 18 `_SD` columns to enable this analysis.")
    else:
        def stability_table(counter, domain_name):
            rows = []
            for subset, count in counter.most_common():
                rows.append({
                    "Domain": domain_name,
                    "Selected subset": " + ".join(subset),
                    "Count": count,
                    "Stability %": 100 * count / n_sim,
                })
            return pd.DataFrame(rows)

        mc_a = stability_table(counts_A, "Domain A")
        mc_b = stability_table(counts_B, "Domain B")
        st.markdown("**Domain A**")
        st.dataframe(mc_a, use_container_width=True, hide_index=True)
        st.markdown("**Domain B**")
        st.dataframe(mc_b, use_container_width=True, hide_index=True)

        if n_sim == 1000 and int(seed) == 42:
            a_ok = counts_A.get(("Acidity",), 0) == 1000
            b_ok = counts_B.get(("Phytate",), 0) == 1000
            if a_ok and b_ok:
                st.success("Validated Monte Carlo check: Acidity = 1000/1000 and Phytate = 1000/1000.")
            else:
                st.error("Validated Monte Carlo check failed for seed 42 / 1000 simulations.")

with tab5:
    st.subheader("Forecasting extension — time-ordered validation")
    st.warning("Forecasting is a separate methodological extension. It is not part of the original MQTF notebook.")
    st.write("The validation uses expanding historical windows: 0–7 → 14, 0–7–14 → 21, and 0–7–14–21 → 28. No random train/test split is used.")

    if not run_forecast_validation:
        st.info("Enable 'Run forecasting extension validation' in the sidebar to calculate MAE, RMSE, R² and MAPE.")
    else:
        bt = backtest(X_mean)
        metric_rows = []
        for variable, metrics in metrics_by_variable(bt).items():
            metric_rows.append({"Variable": variable, **metrics})
        metrics_df = pd.DataFrame(metric_rows)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        st.markdown("### Fold-level predictions")
        st.dataframe(bt, use_container_width=True, hide_index=True)

        st.download_button(
            "Download forecasting validation CSV",
            bt.to_csv(index=False).encode("utf-8"),
            file_name="MQTF_forecasting_backtest.csv",
            mime="text/csv",
        )

with st.expander("Input CSV format"):
    st.write("Required columns: Treatment, Day, followed by all 18 MQTF variables.")
    st.write("For Monte Carlo stability, provide all 18 corresponding `_SD` columns. Either provide all 18 SD columns or provide none.")
    st.code(",".join(["Treatment", "Day"] + FEATURES + [f"{f}_SD" for f in FEATURES]), language="text")

with st.expander("Current data"):
    st.dataframe(df_input, use_container_width=True, hide_index=True)
