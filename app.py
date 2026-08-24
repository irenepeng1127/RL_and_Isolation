import io
import os
import itertools
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator


# ============================================================
# Global chart settings
# ============================================================
TICK_FONTSIZE   = 14
LABEL_FONTSIZE  = 20
TITLE_FONTSIZE  = 20
LEGEND_FONTSIZE = 16
MAX_NAN_GAP = 10

BAND_ITEMS = [
    ("2.4G", 1),
    ("5G", 2),
    ("6G", 3),
    ("6G Japan", 4),
    ("LB", 5),
    ("MB", 6),
    ("CB", 7),
]

BAND_MAPPING = {
    1: (2.4, 2.5),
    2: (5.15, 5.85),
    3: (5.925, 7.125),
    4: (5.925, 6.425),
    5: (0.617, 0.96),
    6: (1.71, 2.7),
    7: (3.3, 4.2),
}


# ============================================================
# Helpers
# ============================================================
def strip_ext_label(name_or_path: str) -> str:
    s = str(name_or_path).strip()
    base = os.path.basename(s)
    root, _ = os.path.splitext(base)
    return root


def sanitize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {c: strip_ext_label(c) for c in df.columns}
    if any(k != v for k, v in rename_map.items()):
        df = df.rename(columns=rename_map)
    return df


def uploaded_to_temp(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1] or ".csv"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    return tmp.name


def find_csv_data_start(file_path):
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            with open(file_path, encoding=enc) as f:
                for i, line in enumerate(f):
                    if "freq" in line.lower() and "," in line:
                        return i, enc
            return 0, enc
        except UnicodeDecodeError:
            continue
    return 0, "latin1"


def find_freq_column(df):
    for col in df.columns:
        if "freq" in str(col).lower():
            return col
    raise ValueError("找不到 Frequency 欄位")


def load_csv_freq_s21_only(path, start_ghz, end_ghz, display_name=None):
    skip, enc = find_csv_data_start(path)
    try:
        df = pd.read_csv(path, skiprows=skip, encoding=enc)
    except Exception:
        df = pd.read_csv(path, skiprows=skip, encoding="latin1")

    if (
        len(df.columns) >= 4
        and str(df.columns[0]).strip() == "Freq(Hz)"
        and str(df.columns[3]).strip() == "S21 Log Mag(dB)"
    ):
        freq = pd.to_numeric(df[df.columns[0]], errors="coerce") / 1e9
        s21 = pd.to_numeric(df[df.columns[3]], errors="coerce")
        sel = (freq >= start_ghz) & (freq <= end_ghz)
        result = pd.DataFrame({
            "Freq": freq[sel],
            "S21 Log Mag(dB)": s21[sel]
        }).dropna()
        result = result.set_index("Freq").sort_index()
        label = strip_ext_label(display_name or path)
        result.columns = [label]
        return result

    return None


def load_csv_files(uploaded_files, start_ghz, end_ghz, is_isolation=False):
    series_list = []
    temp_paths = []

    try:
        for uf in uploaded_files:
            path = uploaded_to_temp(uf)
            temp_paths.append(path)

            if is_isolation:
                special_df = load_csv_freq_s21_only(
                    path, start_ghz, end_ghz, display_name=uf.name
                )
                if special_df is not None:
                    series_list.append(
                        special_df.iloc[:, 0].rename(strip_ext_label(uf.name))
                    )
                    continue

            skip, enc = find_csv_data_start(path)
            try:
                df = pd.read_csv(path, skiprows=skip, encoding=enc)
            except Exception:
                df = pd.read_csv(path, skiprows=skip, encoding="latin1")

            freq_col = find_freq_column(df)
            df[freq_col] = pd.to_numeric(df[freq_col], errors="coerce") / 1e9
            df = df.rename(columns={freq_col: "Freq"}).dropna(subset=["Freq"])
            df = df.set_index("Freq").sort_index()

            df_sel = df.loc[start_ghz:end_ghz]
            df_sel = df_sel[~df_sel.index.duplicated(keep="first")]

            if df_sel.shape[1] < 1:
                continue

            col0 = df_sel.columns[0]
            s = pd.to_numeric(df_sel[col0], errors="coerce")
            s.name = strip_ext_label(uf.name)
            series_list.append(s)

        if not series_list:
            raise ValueError("沒有讀到有效資料。請確認 CSV 格式與頻率範圍。")

        combined = pd.concat(series_list, axis=1, join="outer").sort_index()
        combined.index.name = "Freq"
        return sanitize_df_columns(combined)

    finally:
        for p in temp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass


def read_combined_uploaded(uploaded_file):
    raw = uploaded_file.getvalue()

    last_err = None
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), index_col=0, encoding=enc)
            df.index = pd.to_numeric(df.index, errors="coerce")
            df = df.loc[~df.index.isna()].sort_index()
            df.index.name = "Freq"
            return sanitize_df_columns(df)
        except Exception as e:
            last_err = e

    raise last_err


def is_combined_uploaded(uploaded_file):
    raw = uploaded_file.getvalue()
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            head = pd.read_csv(io.BytesIO(raw), nrows=5, encoding=enc)
            if head.shape[1] >= 2:
                first = str(head.columns[0]).lower()
                return "freq" in first
        except Exception:
            continue
    return False


def resolve_band_codes_to_ranges(band_codes, manual_ranges=None):
    ranges = [BAND_MAPPING[c] for c in band_codes if c in BAND_MAPPING]
    if manual_ranges:
        for a, b in manual_ranges:
            if a is None or b is None:
                continue
            ranges.append((min(a, b), max(a, b)))
    return ranges


def clean_rows_outside_bands(df, band_ranges):
    if not band_ranges:
        return df.copy()

    mask = pd.Series(False, index=df.index)
    for s, e in band_ranges:
        mask |= (df.index >= s) & (df.index <= e)

    df2 = df.copy()
    df2.loc[~mask, :] = np.nan
    return df2


def calculate_vswr(df):
    df = sanitize_df_columns(df)
    vswr = pd.DataFrame(index=df.index)
    for col in df.columns:
        mag = 10 ** (pd.to_numeric(df[col], errors="coerce") / 20)
        with np.errstate(divide="ignore", invalid="ignore"):
            val = (1 + mag) / (1 - mag)
        vswr[strip_ext_label(col)] = val
    return vswr


def _plot_with_gaps(ax, x, y, label=None, color=None):
    finite = np.isfinite(y)
    idx = np.where(finite)[0]
    if idx.size < 2:
        return

    splits = np.where(np.diff(idx) > MAX_NAN_GAP + 1)[0] + 1
    blocks = np.split(idx, splits)
    first = True

    for b in blocks:
        if b.size < 2:
            continue
        line, = ax.plot(
            x[b],
            y[b],
            linestyle="-",
            linewidth=1.8,
            color=color,
            label=label if first else None,
            antialiased=True,
        )
        try:
            line.set_dashes(())
        except Exception:
            pass
        first = False


def make_legend_figure(
    handles,
    labels,
    title="",
    legend_ncol=15,
    legend_fontsize=LEGEND_FONTSIZE,
    max_per_row=10,
    dpi=100,
):
    if not handles or not labels:
        return None

    labels = [strip_ext_label(lb) for lb in labels]
    n = len(labels)
    ncol = min(max_per_row, legend_ncol, n)

    rows = int(np.ceil(n / ncol))
    base_h = 1.0 if title else 0.6
    row_h = 0.8
    height = base_h + rows * row_h

    fig_leg = plt.figure(figsize=(14.4, height), dpi=dpi)
    fig_leg.legend(
        handles,
        labels,
        loc="center",
        ncol=ncol,
        fontsize=legend_fontsize,
        handlelength=1.2,
        columnspacing=0.6,
        labelspacing=0.2,
        borderaxespad=0.5,
    )

    top_margin = 0.92 if title else 0.98
    fig_leg.subplots_adjust(
        left=0.02, right=0.98, top=top_margin, bottom=0.06
    )

    if title:
        fig_leg.suptitle(
            f"{title} Legend",
            fontsize=TITLE_FONTSIZE,
            y=min(0.98, top_margin + 0.02),
        )

    return fig_leg


def plot_filtered_chart_with_bands(
    df,
    title,
    spec=None,
    band_ranges=None,
    xlim=None,
    figsize=(14.4, 9),
    dpi=100,
    legend_fontsize=LEGEND_FONTSIZE,
    legend_ncol=15,
    y_step=None,
    y_min=None,
    y_max=None,
    inline_legend=True,
    y_label=None,
):
    df = sanitize_df_columns(df)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if inline_legend:
        fig.subplots_adjust(
            left=0.08, right=0.98, top=0.90, bottom=0.20
        )
    else:
        fig.subplots_adjust(
            left=0.08, right=0.98, top=0.90, bottom=0.10
        )

    freq_mhz = df.index.to_numpy() * 1000
    colors = itertools.cycle(
        plt.rcParams["axes.prop_cycle"].by_key()["color"]
    )

    for col in df.columns:
        y = pd.to_numeric(df[col], errors="coerce").to_numpy()
        if np.all(np.isnan(y)):
            continue
        _plot_with_gaps(
            ax,
            freq_mhz,
            y,
            label=strip_ext_label(col),
            color=next(colors),
        )

    if spec:
        for s, e, val in spec:
            ax.hlines(
                val,
                s * 1000,
                e * 1000,
                color="red",
                linestyle="--",
            )

    if xlim:
        ax.set_xlim(xlim)

    xmin, xmax = ax.get_xlim()
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.set_xticks(np.arange(np.ceil(xmin / 100) * 100, xmax + 1, 100))

    if y_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_step))

    bottom = y_min if y_min is not None else ax.get_ylim()[0]
    top = y_max if y_max is not None else ax.get_ylim()[1]
    if top <= bottom:
        top = bottom + 0.1
    ax.set_ylim(bottom, top)

    ax.grid(which="major", axis="both", linestyle="-", alpha=0.6)
    ax.tick_params(axis="x", rotation=90, labelsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)

    y0, y1 = ax.get_ylim()
    rects = itertools.cycle(
        plt.rcParams["axes.prop_cycle"].by_key()["color"]
    )

    for s, e in (band_ranges or []):
        ax.add_patch(
            patches.Rectangle(
                (s * 1000, y0),
                (e - s) * 1000,
                y1 - y0,
                color=next(rects),
                alpha=0.25,
            )
        )

    ax.set_title(title, fontsize=TITLE_FONTSIZE)

    if y_label is None:
        t = str(title).lower()
        if "vswr" in t:
            y_label = "VSWR"
        elif "return" in t or "s11" in t:
            y_label = "Return Loss (dB)"
        elif "isolation" in t:
            y_label = "Isolation (dB)"
        else:
            y_label = "Value"

    ax.set_xlabel("Frequency (MHz)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=LABEL_FONTSIZE)

    handles, labels = ax.get_legend_handles_labels()
    labels = [strip_ext_label(lb) for lb in labels]

    if inline_legend:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=legend_ncol,
            fontsize=legend_fontsize,
            handlelength=1.2,
            columnspacing=0.6,
            labelspacing=0.2,
            markerscale=0.6,
            borderaxespad=0.5,
        )
        plt.tight_layout(rect=[0, 0.025, 1, 1])
    else:
        plt.tight_layout()

    return fig, handles, labels


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def df_to_csv_bytes(df):
    return df.to_csv(float_format="%.6f").encode("utf-8-sig")


def parse_optional_float(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    return float(value)


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(
    page_title="RL / VSWR / Isolation",
    page_icon="📡",
    layout="wide",
)

st.title("📡 RL / VSWR / Isolation")

st.caption(
    "Upload CSV files, set frequency / SPEC / band / Y-axis parameters, "
    "preview plots, and download PNG or combined CSV."
)

# Session state
for key, default in {
    "result_df": None,
    "cleaned_df": None,
    "fig_s11": None,
    "fig_vswr": None,
    "fig_iso": None,
    "legend_s11": None,
    "legend_vswr": None,
    "legend_iso": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# 1. Mode
# ============================================================
st.subheader("1. 模式")

mode = st.radio(
    "選擇模式",
    ["Return Loss / VSWR", "Isolation"],
    horizontal=True,
)

if mode == "Return Loss / VSWR":
    c1, c2 = st.columns(2)
    with c1:
        do_s11 = st.checkbox("Return Loss", value=True)
    with c2:
        do_vswr = st.checkbox("VSWR", value=True)
    do_iso = False
else:
    do_s11 = False
    do_vswr = False
    do_iso = True

separate_legend = st.checkbox(
    "獨立顯示圖例",
    value=False,
)


# ============================================================
# 2. Upload
# ============================================================
st.subheader("2. 上傳 CSV")

uploaded_files = st.file_uploader(
    "可一次上傳一個或多個 CSV",
    type=["csv"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.caption(
        "已選擇：" + "、".join([f.name for f in uploaded_files])
    )


# ============================================================
# 3. Frequency range
# ============================================================
st.subheader("3. 頻率範圍 (GHz)")

fc1, fc2 = st.columns(2)

with fc1:
    fstart = st.number_input(
        "Start (GHz)",
        value=0.617,
        format="%.3f",
    )

with fc2:
    fend = st.number_input(
        "End (GHz)",
        value=7.125,
        format="%.3f",
    )


# ============================================================
# 4. SPEC
# ============================================================
st.subheader("4. SPEC 分段")

st.caption("最多 5 組，格式為 Start GHz / End GHz / dB。")

spec_rows = []

for i in range(5):
    c1, c2, c3 = st.columns(3)

    with c1:
        s = st.text_input(
            f"SPEC {i+1} Start",
            key=f"spec_start_{i}",
        )

    with c2:
        e = st.text_input(
            f"SPEC {i+1} End",
            key=f"spec_end_{i}",
        )

    with c3:
        v = st.text_input(
            f"SPEC {i+1} dB",
            key=f"spec_val_{i}",
        )

    if s.strip() and e.strip() and v.strip():
        spec_rows.append((s.strip(), e.strip(), v.strip()))


# ============================================================
# 5. Band
# ============================================================
st.subheader("5. 頻段")

band_name_to_code = {name: code for name, code in BAND_ITEMS}

selected_band_names = st.multiselect(
    "預設頻段",
    options=list(band_name_to_code.keys()),
)

st.caption("也可以另外輸入最多 5 組手動頻段。")

manual_ranges_text = []

for i in range(5):
    c1, c2 = st.columns(2)

    with c1:
        s = st.text_input(
            f"Manual {i+1} Start (GHz)",
            key=f"manual_start_{i}",
        )

    with c2:
        e = st.text_input(
            f"Manual {i+1} End (GHz)",
            key=f"manual_end_{i}",
        )

    if s.strip() and e.strip():
        manual_ranges_text.append((s.strip(), e.strip()))


# ============================================================
# 6. Y axis
# ============================================================
st.subheader("6. Y 軸設定")

y1, y2, y3 = st.columns(3)

with y1:
    st.markdown("**Return Loss**")
    s11_ymin = st.text_input("Ymin", key="s11_ymin")
    s11_ymax = st.text_input("Ymax", key="s11_ymax")
    s11_ystep = st.text_input("Step", value="5", key="s11_ystep")

with y2:
    st.markdown("**VSWR**")
    vswr_ymin = st.text_input("Ymin", value="1", key="vswr_ymin")
    vswr_ymax = st.text_input("Ymax", key="vswr_ymax")
    vswr_ystep = st.text_input("Step", value="1", key="vswr_ystep")

with y3:
    st.markdown("**Isolation**")
    iso_ymin = st.text_input("Ymin", key="iso_ymin")
    iso_ymax = st.text_input("Ymax", key="iso_ymax")
    iso_ystep = st.text_input("Step", value="5", key="iso_ystep")


# ============================================================
# 7. Cleaning
# ============================================================
st.subheader("7. 資料範圍")

clean_non_band = st.checkbox(
    "清除非頻段範圍",
    value=False,
    help="勾選後，未落在所選 Band / Manual Band 的資料會變成空值。",
)


# ============================================================
# Run
# ============================================================
run = st.button(
    "執行預覽",
    type="primary",
    use_container_width=True,
)

if run:
    st.session_state.fig_s11 = None
    st.session_state.fig_vswr = None
    st.session_state.fig_iso = None
    st.session_state.legend_s11 = None
    st.session_state.legend_vswr = None
    st.session_state.legend_iso = None
    st.session_state.cleaned_df = None

    try:
        if not uploaded_files:
            raise ValueError("請先上傳 CSV。")

        if fstart >= fend:
            raise ValueError("Start 必須小於 End。")

        if not do_s11 and not do_vswr and not do_iso:
            raise ValueError("請至少選擇一個模式。")

        # SPEC
        specs = []
        for s, e, v in spec_rows:
            specs.append((float(s), float(e), float(v)))

        # manual ranges
        manual_ranges = []
        for s, e in manual_ranges_text:
            manual_ranges.append((float(s), float(e)))

        selected_band_codes = [
            band_name_to_code[n] for n in selected_band_names
        ]
        band_ranges = resolve_band_codes_to_ranges(
            selected_band_codes,
            manual_ranges,
        )

        # Y
        s11_ymin_f = parse_optional_float(s11_ymin)
        s11_ymax_f = parse_optional_float(s11_ymax)
        s11_ystep_f = parse_optional_float(s11_ystep)
        vswr_ymin_f = parse_optional_float(vswr_ymin)
        vswr_ymax_f = parse_optional_float(vswr_ymax)
        vswr_ystep_f = parse_optional_float(vswr_ystep)
        iso_ymin_f = parse_optional_float(iso_ymin)
        iso_ymax_f = parse_optional_float(iso_ymax)
        iso_ystep_f = parse_optional_float(iso_ystep)

        s11_ystep_f = 5 if s11_ystep_f is None else s11_ystep_f
        vswr_ystep_f = 1 if vswr_ystep_f is None else vswr_ystep_f
        iso_ystep_f = 5 if iso_ystep_f is None else iso_ystep_f

        if s11_ystep_f <= 0 or vswr_ystep_f <= 0 or iso_ystep_f <= 0:
            raise ValueError("Y 軸 Step 必須大於 0。")

        # combined input
        if len(uploaded_files) == 1 and is_combined_uploaded(uploaded_files[0]):
            df_combined = read_combined_uploaded(uploaded_files[0])
        else:
            df_combined = load_csv_files(
                uploaded_files,
                fstart,
                fend,
                is_isolation=do_iso,
            )

        df_combined = sanitize_df_columns(df_combined)

        source_df = df_combined.loc[
            (df_combined.index >= fstart)
            & (df_combined.index <= fend)
        ].copy()

        if source_df.empty:
            raise ValueError("指定頻率範圍內沒有資料。")

        if clean_non_band:
            working_df = clean_rows_outside_bands(
                source_df,
                band_ranges,
            )
            st.session_state.cleaned_df = working_df.copy()
        else:
            working_df = source_df.copy()

        st.session_state.result_df = df_combined.copy()

        inline_legend = not separate_legend

        # Convert RL spec to VSWR spec
        seg_vswr = []
        for s, e, v in specs:
            mag = 10 ** (v / 20)
            if mag >= 1:
                raise ValueError(
                    f"SPEC {v} dB 無法轉成有效 VSWR。Return Loss dB 應小於 0。"
                )
            seg_vswr.append(
                (s, e, (1 + mag) / (1 - mag))
            )

        if do_iso:
            fig, h, lb = plot_filtered_chart_with_bands(
                working_df,
                "Isolation",
                specs,
                band_ranges,
                xlim=(fstart * 1000, fend * 1000),
                y_step=iso_ystep_f,
                y_min=iso_ymin_f,
                y_max=iso_ymax_f,
                inline_legend=inline_legend,
                y_label="Isolation (dB)",
            )
            st.session_state.fig_iso = fig

            if separate_legend:
                st.session_state.legend_iso = make_legend_figure(
                    h,
                    lb,
                    title="Isolation",
                    legend_ncol=15,
                    legend_fontsize=LEGEND_FONTSIZE,
                )

        else:
            if do_s11:
                fig, h, lb = plot_filtered_chart_with_bands(
                    working_df,
                    "Return Loss",
                    specs,
                    band_ranges,
                    xlim=(fstart * 1000, fend * 1000),
                    y_step=s11_ystep_f,
                    y_min=s11_ymin_f,
                    y_max=s11_ymax_f,
                    inline_legend=inline_legend,
                    y_label="Return Loss (dB)",
                )
                st.session_state.fig_s11 = fig

                if separate_legend:
                    st.session_state.legend_s11 = make_legend_figure(
                        h,
                        lb,
                        title="Return Loss",
                        legend_ncol=15,
                        legend_fontsize=LEGEND_FONTSIZE,
                    )

            if do_vswr:
                vswr_df = calculate_vswr(working_df)

                fig, h, lb = plot_filtered_chart_with_bands(
                    vswr_df,
                    "VSWR",
                    seg_vswr,
                    band_ranges,
                    xlim=(fstart * 1000, fend * 1000),
                    y_step=vswr_ystep_f,
                    y_min=vswr_ymin_f if vswr_ymin_f is not None else 1,
                    y_max=vswr_ymax_f,
                    inline_legend=inline_legend,
                    y_label="VSWR",
                )
                st.session_state.fig_vswr = fig

                if separate_legend:
                    st.session_state.legend_vswr = make_legend_figure(
                        h,
                        lb,
                        title="VSWR",
                        legend_ncol=15,
                        legend_fontsize=LEGEND_FONTSIZE,
                    )

        st.success("分析完成。")

    except Exception as e:
        st.error(f"執行失敗：{e}")


# ============================================================
# Output
# ============================================================
if any([
    st.session_state.fig_s11 is not None,
    st.session_state.fig_vswr is not None,
    st.session_state.fig_iso is not None,
]):
    st.divider()
    st.header("結果")

if st.session_state.fig_s11 is not None:
    st.subheader("Return Loss")
    st.pyplot(st.session_state.fig_s11, use_container_width=True)
    st.download_button(
        "下載 Return Loss PNG",
        data=fig_to_png_bytes(st.session_state.fig_s11),
        file_name="s11.png",
        mime="image/png",
        use_container_width=True,
    )

    if st.session_state.legend_s11 is not None:
        st.caption("Return Loss Legend")
        st.pyplot(st.session_state.legend_s11, use_container_width=True)
        st.download_button(
            "下載 Return Loss Legend PNG",
            data=fig_to_png_bytes(st.session_state.legend_s11),
            file_name="legend_s11.png",
            mime="image/png",
            use_container_width=True,
        )

if st.session_state.fig_vswr is not None:
    st.subheader("VSWR")
    st.pyplot(st.session_state.fig_vswr, use_container_width=True)
    st.download_button(
        "下載 VSWR PNG",
        data=fig_to_png_bytes(st.session_state.fig_vswr),
        file_name="vswr.png",
        mime="image/png",
        use_container_width=True,
    )

    if st.session_state.legend_vswr is not None:
        st.caption("VSWR Legend")
        st.pyplot(st.session_state.legend_vswr, use_container_width=True)
        st.download_button(
            "下載 VSWR Legend PNG",
            data=fig_to_png_bytes(st.session_state.legend_vswr),
            file_name="legend_vswr.png",
            mime="image/png",
            use_container_width=True,
        )

if st.session_state.fig_iso is not None:
    st.subheader("Isolation")
    st.pyplot(st.session_state.fig_iso, use_container_width=True)
    st.download_button(
        "下載 Isolation PNG",
        data=fig_to_png_bytes(st.session_state.fig_iso),
        file_name="iso.png",
        mime="image/png",
        use_container_width=True,
    )

    if st.session_state.legend_iso is not None:
        st.caption("Isolation Legend")
        st.pyplot(st.session_state.legend_iso, use_container_width=True)
        st.download_button(
            "下載 Isolation Legend PNG",
            data=fig_to_png_bytes(st.session_state.legend_iso),
            file_name="legend_isolation.png",
            mime="image/png",
            use_container_width=True,
        )


if st.session_state.result_df is not None:
    st.divider()
    st.subheader("Combined CSV")

    export_df = (
        st.session_state.cleaned_df
        if st.session_state.cleaned_df is not None
        else st.session_state.result_df
    )

    st.download_button(
        "下載 combined.csv",
        data=df_to_csv_bytes(export_df),
        file_name="combined.csv",
        mime="text/csv",
        use_container_width=True,
    )
