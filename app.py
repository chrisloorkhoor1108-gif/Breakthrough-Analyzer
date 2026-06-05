import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO

st.set_page_config(page_title="Breakthrough Curve Analyzer", layout="wide")

st.title("Breakthrough Curve Analyzer")
st.write("Upload a mass spec `.txt` file exported from Process Eye / Spectra International.")

uploaded_file = st.file_uploader("Upload your mass spec TXT file", type=["txt"])


def read_mass_spec_txt(uploaded_file):
    """
    Reads a Process Eye / Spectra International TXT export.
    Finds the [Scan Data] section and loads it as a pandas DataFrame.
    """

    raw_text = uploaded_file.read().decode("utf-8", errors="replace")
    lines = raw_text.splitlines()

    scan_start = None
    for i, line in enumerate(lines):
        if "[Scan Data" in line:
            scan_start = i + 1
            break

    if scan_start is None:
        raise ValueError("Could not find the [Scan Data] section in this file.")

    table_lines = []
    for line in lines[scan_start:]:
        stripped = line.strip()

        if stripped.startswith('"[') or stripped.startswith("["):
            break

        if stripped:
            table_lines.append(line)

    if not table_lines:
        raise ValueError("Found [Scan Data], but could not read any table rows.")

    table_text = "\n".join(table_lines)

    df = pd.read_csv(StringIO(table_text), sep="\t", quotechar='"')
    df = df.dropna(axis=1, how="all")
    df.columns = [str(col).strip().replace('"', "") for col in df.columns]

    for col in df.columns:
        if col != "Time":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        df = df.dropna(subset=["Time"])
        df["Elapsed Time (min)"] = (
            df["Time"] - df["Time"].iloc[0]
        ).dt.total_seconds() / 60

    return df


def normalize_signal(df, signal_col, baseline_range, c0_range):
    """
    Normalizes a selected mass spec signal using:
    normalized = (signal - baseline) / (c0 - baseline)
    """

    baseline_mask = (
        (df["Elapsed Time (min)"] >= baseline_range[0]) &
        (df["Elapsed Time (min)"] <= baseline_range[1])
    )

    c0_mask = (
        (df["Elapsed Time (min)"] >= c0_range[0]) &
        (df["Elapsed Time (min)"] <= c0_range[1])
    )

    baseline = df.loc[baseline_mask, signal_col].mean()
    c0 = df.loc[c0_mask, signal_col].mean()

    if pd.isna(baseline) or pd.isna(c0):
        raise ValueError(f"Could not calculate baseline or C₀ for {signal_col}.")

    if c0 == baseline:
        raise ValueError(f"C₀ and baseline are equal for {signal_col}, so it cannot be normalized.")

    normalized = (df[signal_col] - baseline) / (c0 - baseline)

    return normalized, baseline, c0


def find_threshold_time(df, threshold):
    """
    Finds the first elapsed time where CO2 C/C0 reaches or exceeds a threshold.
    """
    crossed = df[df["CO2 C/C0"] >= threshold]

    if crossed.empty:
        return None

    return crossed["Elapsed Time (min)"].iloc[0]


if uploaded_file is not None:
    try:
        df = read_mass_spec_txt(uploaded_file)

        st.success("File loaded successfully.")

        st.subheader("Data Preview")
        st.dataframe(df.head())

        st.subheader("Select Signal to Plot")

        mass_columns = [col for col in df.columns if col.startswith("Mass")]

        other_numeric_columns = [
            col for col in df.select_dtypes(include="number").columns
            if col not in mass_columns
        ]

        y_options = mass_columns + other_numeric_columns

        x_col = st.selectbox(
            "X-axis",
            ["Elapsed Time (min)", "Scan"] if "Elapsed Time (min)" in df.columns else df.columns
        )

        y_col = st.selectbox(
            "Y-axis / Mass signal",
            y_options,
            index=y_options.index("Mass 44") if "Mass 44" in y_options else 0
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df[x_col], df[y_col])
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} vs {x_col}")
        ax.grid(True)

        st.pyplot(fig)

        st.subheader("Basic Signal Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Minimum Signal", f"{df[y_col].min():.3g}")

        with col2:
            st.metric("Maximum Signal", f"{df[y_col].max():.3g}")

        with col3:
            st.metric("Number of Data Points", len(df))

        # ---------------------------------------------------------
        # NORMALIZED BREAKTHROUGH CURVE SECTION
        # ---------------------------------------------------------

        st.subheader("Normalize Breakthrough Curves")

        if "Mass 44" in df.columns and "Elapsed Time (min)" in df.columns:
            col1, col2 = st.columns(2)

            with col1:
                co2_signal_col = st.selectbox(
                    "Select CO₂ signal column",
                    mass_columns,
                    index=mass_columns.index("Mass 44") if "Mass 44" in mass_columns else 0
                )

            with col2:
                n2_signal_col = st.selectbox(
                    "Select N₂ / inert tracer signal column",
                    mass_columns,
                    index=mass_columns.index("Mass 28") if "Mass 28" in mass_columns else 0
                )

            max_time = float(df["Elapsed Time (min)"].max())

            st.write("Choose time ranges for baseline and final outlet concentration.")

            baseline_range = st.slider(
                "Baseline region before breakthrough, min",
                min_value=0.0,
                max_value=max_time,
                value=(0.0, min(1.0, max_time)),
                step=0.1
            )

            c0_range = st.slider(
                "Final steady C₀ region, min",
                min_value=0.0,
                max_value=max_time,
                value=(max(0.0, max_time - 1.0), max_time),
                step=0.1
            )

            try:
                df["CO2 C/C0"], co2_baseline, co2_c0 = normalize_signal(
                    df,
                    co2_signal_col,
                    baseline_range,
                    c0_range
                )

                df["N2 C/C0"], n2_baseline, n2_c0 = normalize_signal(
                    df,
                    n2_signal_col,
                    baseline_range,
                    c0_range
                )

                df["CO2 C/C0 Plot"] = df["CO2 C/C0"].clip(lower=0, upper=1.2)
                df["N2 C/C0 Plot"] = df["N2 C/C0"].clip(lower=0, upper=1.2)

                # Baseline-corrected CO2-only method:
                # area = ∫(1 - CO2 C/C0) dt
                df["CO2 Adsorbed Fraction"] = (1 - df["CO2 C/C0"]).clip(lower=0, upper=1)

                # N2 tracer method:
                # area = ∫(N2 normalized curve - CO2 normalized curve) dt
                df["N2-CO2 Difference"] = (df["N2 C/C0"] - df["CO2 C/C0"]).clip(lower=0)

                t_05 = find_threshold_time(df, 0.05)
                t_50 = find_threshold_time(df, 0.50)
                t_95 = find_threshold_time(df, 0.95)

                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"CO₂ baseline = `{co2_baseline:.3g}`")
                    st.write(f"CO₂ C₀ = `{co2_c0:.3g}`")

                with col2:
                    st.write(f"N₂ baseline = `{n2_baseline:.3g}`")
                    st.write(f"N₂ C₀ = `{n2_c0:.3g}`")

                show_threshold_lines = st.checkbox(
                    "Show 5%, 50%, and 95% breakthrough lines",
                    value=False
                )

                show_n2_curve = st.checkbox(
                    "Show normalized N₂ curve on breakthrough plot",
                    value=True
                )

                fig2, ax2 = plt.subplots(figsize=(10, 5))

                ax2.plot(
                    df["Elapsed Time (min)"],
                    df["CO2 C/C0 Plot"],
                    label=f"CO₂ baseline-corrected: {co2_signal_col}"
                )

                if show_n2_curve:
                    ax2.plot(
                        df["Elapsed Time (min)"],
                        df["N2 C/C0 Plot"],
                        label=f"N₂ / tracer: {n2_signal_col}"
                    )

                if show_threshold_lines:
                    ax2.axhline(0.05, linestyle="--", label="5% breakthrough")
                    ax2.axhline(0.50, linestyle="--", label="50% breakthrough")
                    ax2.axhline(0.95, linestyle="--", label="95% saturation")

                    if t_05 is not None:
                        ax2.axvline(t_05, linestyle="--")
                    if t_50 is not None:
                        ax2.axvline(t_50, linestyle="--")
                    if t_95 is not None:
                        ax2.axvline(t_95, linestyle="--")

                ax2.set_xlabel("Elapsed Time (min)")
                ax2.set_ylabel("Normalized signal")
                ax2.set_title("Normalized Breakthrough Curves")
                ax2.grid(True)
                ax2.legend()

                st.pyplot(fig2)

                st.subheader("Breakthrough Time Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    if t_05 is not None:
                        st.metric("5% Breakthrough Time", f"{t_05:.3f} min")
                    else:
                        st.metric("5% Breakthrough Time", "Not reached")

                with col2:
                    if t_50 is not None:
                        st.metric("50% Breakthrough Time", f"{t_50:.3f} min")
                    else:
                        st.metric("50% Breakthrough Time", "Not reached")

                with col3:
                    if t_95 is not None:
                        st.metric("95% Saturation Time", f"{t_95:.3f} min")
                    else:
                        st.metric("95% Saturation Time", "Not reached")

                # ---------------------------------------------------------
                # CAPACITY CALCULATION SECTION
                # ---------------------------------------------------------

                st.subheader("CO₂ Adsorption Capacity Calculation")

                capacity_method = st.selectbox(
                    "Capacity calculation method",
                    [
                        "Excel-style CO₂ in/out method",
                        "Baseline-corrected CO₂-only area method",
                        "Area between N₂ and CO₂ curves"
                    ],
                    index=0
                )

                if capacity_method == "Excel-style CO₂ in/out method":
                    st.write(
                        "Capacity is calculated using your spreadsheet-style approach: "
                        "`CO₂ adsorbed = CO₂ in - CO₂ out`, where "
                        "`C/C₀ = raw CO₂ signal / CO₂eq`."
                    )
                elif capacity_method == "Baseline-corrected CO₂-only area method":
                    st.write(
                        "Capacity is calculated using "
                        "`∫(1 - baseline-corrected CO₂ C/C₀) dt`."
                    )
                else:
                    st.write(
                        "Capacity is calculated using "
                        "`∫(N₂ normalized curve - CO₂ normalized curve) dt`."
                    )

                col1, col2, col3 = st.columns(3)

                with col1:
                    total_flow_sccm = st.number_input(
                        "Total gas flow rate, sccm",
                        min_value=0.0,
                        value=50.0,
                        step=1.0
                    )

                with col2:
                    co2_percent = st.number_input(
                        "Inlet CO₂ concentration, %",
                        min_value=0.0,
                        max_value=100.0,
                        value=10.0,
                        step=1.0
                    )

                with col3:
                    sample_mass_mg = st.number_input(
                        "Sample mass, mg",
                        min_value=0.0,
                        value=50.0,
                        step=0.1
                    )

                molar_volume_ml_per_mol = st.number_input(
                    "Molar volume used for sccm conversion, mL/mol",
                    min_value=1.0,
                    value=47000.0,
                    step=1.0,
                    help=(
                        "Use 47000 mL/mol if that matches your experimental flow conversion. "
                        "Use 22414 mL/mol for standard molar volume near STP."
                    )
                )

                if capacity_method == "Excel-style CO₂ in/out method":
                    co2eq_signal = st.number_input(
                        "CO₂eq signal value for Excel-style C/C₀",
                        min_value=0.0000001,
                        value=float(co2_c0),
                        step=0.1,
                        help=(
                            "This is the CO₂ equilibrium/final signal used in your spreadsheet. "
                            "For example, if your sheet used CO₂eq = 95.9, enter 95.9 here."
                        )
                    )

                    st.caption(
                        f"Excel-style C/C₀ will be calculated as `{co2_signal_col} / {co2eq_signal:.3g}`."
                    )

                st.write("Choose when adsorption starts and ends.")

                col1, col2 = st.columns(2)

                with col1:
                    adsorption_start_time = st.number_input(
                        "Adsorption start time, min",
                        min_value=0.0,
                        max_value=max_time,
                        value=0.0,
                        step=0.1,
                        help=(
                            "Set this to the time when CO₂ actually started entering the bed. "
                            "Time before this will not count toward capacity."
                        )
                    )

                with col2:
                    adsorption_end_time = st.number_input(
                        "Adsorption end time, min",
                        min_value=adsorption_start_time,
                        max_value=max_time,
                        value=max_time,
                        step=0.1,
                        help=(
                            "Set this to the time when adsorption ended. "
                            "Time after this will not count toward capacity."
                        )
                    )

                integration_range = st.slider(
                    "Fine-tune capacity integration range, min",
                    min_value=adsorption_start_time,
                    max_value=adsorption_end_time,
                    value=(adsorption_start_time, adsorption_end_time),
                    step=0.1
                )

                integration_mask = (
                    (df["Elapsed Time (min)"] >= integration_range[0]) &
                    (df["Elapsed Time (min)"] <= integration_range[1])
                )

                integration_df = df.loc[integration_mask].copy()

                # Correct time so that adsorption start becomes t = 0 for capacity calculation.
                integration_df["Corrected Time (min)"] = (
                    integration_df["Elapsed Time (min)"] - adsorption_start_time
                )

                if len(integration_df) < 2:
                    st.warning("Not enough data points in the selected integration range.")
                elif total_flow_sccm <= 0:
                    st.warning("Total flow rate must be greater than 0.")
                elif co2_percent <= 0:
                    st.warning("CO₂ concentration must be greater than 0.")
                elif sample_mass_mg <= 0:
                    st.warning("Sample mass must be greater than 0.")
                elif adsorption_end_time <= adsorption_start_time:
                    st.warning("Adsorption end time must be after adsorption start time.")
                else:
                    # -----------------------------------------------------
                    # Molar flow calculation
                    # -----------------------------------------------------
                    # First calculate TOTAL molar flow from total gas flow.
                    # sccm = mL/min at the selected reference condition.
                    total_mol_per_min = total_flow_sccm / molar_volume_ml_per_mol

                    # Then calculate CO2 inlet molar flow as the CO2 fraction
                    # of the total molar flow.
                    co2_fraction = co2_percent / 100
                    co2_mol_per_min = total_mol_per_min * co2_fraction
                    co2_mol_per_sec = co2_mol_per_min / 60

                    # Equivalent CO2 sccm is included only for clarity.
                    equivalent_co2_sccm = total_flow_sccm * co2_fraction

                    sample_mass_g = sample_mass_mg / 1000

                    if capacity_method == "Excel-style CO₂ in/out method":
                        # -------------------------------------------------
                        # Excel-style method
                        # -------------------------------------------------
                        # This follows the spreadsheet approach:
                        # C/C0 = raw CO2 signal / CO2eq
                        # CO2 out per interval = inlet CO2 mol/s * C/C0 * delta_t
                        # CO2 in = inlet CO2 mol/s * total adsorption time
                        # CO2 adsorbed = CO2 in - CO2 out
                        # -------------------------------------------------

                        integration_df["Excel C/C0"] = (
                            integration_df[co2_signal_col] / co2eq_signal
                        )

                        # Prevent obviously nonphysical negative outlet fractions.
                        # Values above 1 are kept because some real signals/noise can overshoot.
                        integration_df["Excel C/C0"] = integration_df["Excel C/C0"].clip(lower=0)

                        corrected_time_sec = (
                            integration_df["Corrected Time (min)"].to_numpy() * 60
                        )

                        c_over_c0 = integration_df["Excel C/C0"].to_numpy()

                        # Use actual time spacing between data points.
                        dt_sec = np.diff(corrected_time_sec)

                        # Use left-endpoint rectangle method to mimic row-by-row Excel summation.
                        c_over_c0_for_intervals = c_over_c0[:-1]

                        adsorption_duration_sec = corrected_time_sec[-1] - corrected_time_sec[0]

                        total_co2_in_mol = co2_mol_per_sec * adsorption_duration_sec
                        co2_out_mol = np.sum(co2_mol_per_sec * c_over_c0_for_intervals * dt_sec)
                        adsorbed_mol = total_co2_in_mol - co2_out_mol

                        # Equivalent area in minutes for comparison with area methods.
                        area_min = adsorbed_mol / co2_mol_per_min

                        capacity_mmol_g = (adsorbed_mol * 1000) / sample_mass_g

                        method_details = (
                            f"Total CO₂ in = `{total_co2_in_mol:.3e} mol`  \n"
                            f"CO₂ out = `{co2_out_mol:.3e} mol`  \n"
                            f"CO₂ adsorbed = `{adsorbed_mol:.3e} mol`"
                        )

                        y_label = "Raw CO₂ C/C₀"
                        plot_signal = integration_df["Excel C/C0"]

                    else:
                        # -------------------------------------------------
                        # Area-based methods
                        # -------------------------------------------------

                        time_min = integration_df["Corrected Time (min)"].to_numpy()

                        if capacity_method == "Area between N₂ and CO₂ curves":
                            integration_signal = integration_df["N2-CO2 Difference"].to_numpy()
                            y_label = "N₂ - CO₂ normalized difference"
                        else:
                            integration_signal = integration_df["CO2 Adsorbed Fraction"].to_numpy()
                            y_label = "CO₂ adsorbed fraction"

                        # Area has units of minutes because the integrated signal is dimensionless.
                        area_min = np.trapezoid(integration_signal, time_min)

                        adsorbed_mol = co2_mol_per_min * area_min
                        capacity_mmol_g = (adsorbed_mol * 1000) / sample_mass_g

                        method_details = (
                            f"CO₂ adsorbed = `{adsorbed_mol:.3e} mol`"
                        )

                    st.info(
                        f"Using total gas flow = {total_flow_sccm:.3g} sccm. "
                        f"Total molar flow = {total_mol_per_min:.3e} mol/min. "
                        f"CO₂ fraction = {co2_fraction:.3g}. "
                        f"CO₂ inlet molar flow = {co2_mol_per_min:.3e} mol/min."
                    )

                    st.subheader("Capacity Results")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Equivalent Integrated Area", f"{area_min:.3f} min")

                    with col2:
                        st.metric("Adsorbed CO₂", f"{adsorbed_mol * 1000:.3f} mmol")

                    with col3:
                        st.metric("Capacity", f"{capacity_mmol_g:.3f} mmol/g")

                    st.write(f"Total molar flow: `{total_mol_per_min:.3e} mol/min`")
                    st.write(f"CO₂ inlet molar flow: `{co2_mol_per_min:.3e} mol/min`")
                    st.write(f"Adsorption start: `{adsorption_start_time:.3f} min`")
                    st.write(f"Adsorption end: `{adsorption_end_time:.3f} min`")
                    st.write(
                        f"Integrated duration: "
                        f"`{integration_range[1] - integration_range[0]:.3f} min`"
                    )
                    st.write(method_details)

                    st.caption(
                        f"Equivalent CO₂ flow = {equivalent_co2_sccm:.3g} sccm "
                        f"from {total_flow_sccm:.3g} sccm total gas at {co2_percent:.3g}% CO₂."
                    )

                    fig3, ax3 = plt.subplots(figsize=(10, 5))

                    if capacity_method == "Excel-style CO₂ in/out method":
                        ax3.plot(
                            df["Elapsed Time (min)"],
                            (df[co2_signal_col] / co2eq_signal).clip(lower=0),
                            label=f"Excel-style CO₂ C/C₀: {co2_signal_col}/{co2eq_signal:.3g}"
                        )

                        ax3.fill_between(
                            integration_df["Elapsed Time (min)"],
                            integration_df["Excel C/C0"],
                            1,
                            where=(integration_df["Excel C/C0"] <= 1),
                            alpha=0.3,
                            label="Adsorbed fraction area"
                        )

                    elif capacity_method == "Area between N₂ and CO₂ curves":
                        ax3.plot(
                            df["Elapsed Time (min)"],
                            df["N2 C/C0 Plot"],
                            label=f"N₂ / tracer: {n2_signal_col}"
                        )

                        ax3.plot(
                            df["Elapsed Time (min)"],
                            df["CO2 C/C0 Plot"],
                            label=f"CO₂: {co2_signal_col}"
                        )

                        ax3.fill_between(
                            integration_df["Elapsed Time (min)"],
                            integration_df["CO2 C/C0"].clip(lower=0, upper=1.2),
                            integration_df["N2 C/C0"].clip(lower=0, upper=1.2),
                            where=(
                                integration_df["N2 C/C0"] >= integration_df["CO2 C/C0"]
                            ),
                            alpha=0.3,
                            label="Integrated area between curves"
                        )

                    else:
                        ax3.plot(
                            df["Elapsed Time (min)"],
                            df["CO2 Adsorbed Fraction"],
                            label="1 - baseline-corrected CO₂ C/C₀"
                        )

                        ax3.fill_between(
                            integration_df["Elapsed Time (min)"],
                            integration_df["CO2 Adsorbed Fraction"],
                            alpha=0.3,
                            label="Integrated area"
                        )

                    ax3.axvline(
                        adsorption_start_time,
                        linestyle="--",
                        label="Adsorption start"
                    )

                    ax3.axvline(
                        adsorption_end_time,
                        linestyle="--",
                        label="Adsorption end"
                    )

                    ax3.set_xlabel("Elapsed Time (min)")
                    ax3.set_ylabel(y_label)
                    ax3.set_title("Capacity Integration Area")
                    ax3.grid(True)
                    ax3.legend()

                    st.pyplot(fig3)

                    st.info(
                        "The Excel-style method is intended to match your spreadsheet calculation. "
                        "If it still differs, check that the CO₂eq value, adsorption start time, "
                        "adsorption end time, sample mass, molar volume, and gas composition match "
                        "the spreadsheet exactly."
                    )

            except Exception as norm_error:
                st.error(f"Something went wrong during normalization: {norm_error}")

        else:
            st.warning("Mass 44 or elapsed time was not found in this file.")

        # ---------------------------------------------------------
        # DOWNLOAD CLEANED DATA
        # ---------------------------------------------------------

        st.subheader("Download Cleaned Data")

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download cleaned CSV",
            data=csv,
            file_name="cleaned_mass_spec_data.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Something went wrong while reading the file: {e}")
