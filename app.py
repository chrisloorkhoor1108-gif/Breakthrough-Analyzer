import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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

    # Find the scan data section
    scan_start = None
    for i, line in enumerate(lines):
        if "[Scan Data" in line:
            scan_start = i + 1  # header row is the next line
            break

    if scan_start is None:
        raise ValueError("Could not find the [Scan Data] section in this file.")

    # Collect table lines until the next section begins
    table_lines = []
    for line in lines[scan_start:]:
        stripped = line.strip()

        # Stop when another bracketed section starts
        if stripped.startswith('"[') or stripped.startswith("["):
            break

        # Skip empty lines
        if stripped:
            table_lines.append(line)

    if not table_lines:
        raise ValueError("Found [Scan Data], but could not read any table rows.")

    table_text = "\n".join(table_lines)

    # Read tab-separated table
    df = pd.read_csv(StringIO(table_text), sep="\t", quotechar='"')

    # Remove empty columns caused by trailing tabs
    df = df.dropna(axis=1, how="all")

    # Clean column names
    df.columns = [str(col).strip().replace('"', "") for col in df.columns]

    # Convert numeric columns where possible
    for col in df.columns:
        if col != "Time":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert time column
    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        df = df.dropna(subset=["Time"])
        df["Elapsed Time (min)"] = (
            df["Time"] - df["Time"].iloc[0]
        ).dt.total_seconds() / 60

    return df


def find_threshold_time(df, threshold):
    """
    Finds the first elapsed time where C/C0 reaches or exceeds a threshold.
    """
    crossed = df[df["C/C0"] >= threshold]

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
            st.metric("Minimum Signal", f"{df[y_col].min():.4g}")

        with col2:
            st.metric("Maximum Signal", f"{df[y_col].max():.4g}")

        with col3:
            st.metric("Number of Data Points", len(df))

        # ---------------------------------------------------------
        # NORMALIZED BREAKTHROUGH CURVE SECTION
        # ---------------------------------------------------------

        st.subheader("Normalize Breakthrough Curve")

        if "Mass 44" in df.columns and "Elapsed Time (min)" in df.columns:
            signal_col = st.selectbox(
                "Select CO₂ signal column for normalization",
                mass_columns,
                index=mass_columns.index("Mass 44") if "Mass 44" in mass_columns else 0
            )

            max_time = float(df["Elapsed Time (min)"].max())

            st.write("Choose time ranges for baseline and final outlet concentration.")

            baseline_range = st.slider(
                "Baseline region before CO₂ breakthrough, min",
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

            if c0 == baseline:
                st.error("C₀ and baseline are equal, so C/C₀ cannot be calculated.")
            else:
                df["C/C0"] = (df[signal_col] - baseline) / (c0 - baseline)
                df["C/C0"] = df["C/C0"].clip(lower=0, upper=1.2)

                t_05 = find_threshold_time(df, 0.05)
                t_50 = find_threshold_time(df, 0.50)
                t_95 = find_threshold_time(df, 0.95)

                st.write(f"Baseline = `{baseline:.4g}`")
                st.write(f"C₀ = `{c0:.4g}`")

                fig2, ax2 = plt.subplots(figsize=(10, 5))

                ax2.plot(
                    df["Elapsed Time (min)"],
                    df["C/C0"],
                    label="C/C₀"
                )

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
                ax2.set_ylabel("C/C₀")
                ax2.set_title(f"Normalized Breakthrough Curve: {signal_col}")
                ax2.grid(True)
                ax2.legend()

                st.pyplot(fig2)

                st.subheader("Breakthrough Time Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    if t_05 is not None:
                        st.metric("5% Breakthrough Time", f"{t_05:.2f} min")
                    else:
                        st.metric("5% Breakthrough Time", "Not reached")

                with col2:
                    if t_50 is not None:
                        st.metric("50% Breakthrough Time", f"{t_50:.2f} min")
                    else:
                        st.metric("50% Breakthrough Time", "Not reached")

                with col3:
                    if t_95 is not None:
                        st.metric("95% Saturation Time", f"{t_95:.2f} min")
                    else:
                        st.metric("95% Saturation Time", "Not reached")

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
