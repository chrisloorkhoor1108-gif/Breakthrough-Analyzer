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
        df["Elapsed Time (min)"] = (df["Time"] - df["Time"].iloc[0]).dt.total_seconds() / 60

    return df


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
