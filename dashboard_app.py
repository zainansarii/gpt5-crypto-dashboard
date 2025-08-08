import os
import time
import math
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.express as px
from portfolio_loader import (
    load_raw_sheet,
    extract_transactions,
    current_positions,  # baseline only
    extract_holdings_summary,
    compute_cost_basis_positions,
    reconcile_positions_with_authoritative,
    compute_reconciliation_stats,
    portfolio_invested_timeseries,
    compute_discrepancy_sets,
    average_cost_ledger,
    persist_portfolio_snapshot,
    DUST_THRESHOLD,
)
from price_providers import fetch_prices

st.set_page_config(page_title='Crypto Portfolio Dashboard', page_icon='💹', layout='wide', initial_sidebar_state='expanded')

# ---------------- Style ----------------
CUSTOM_CSS = """
<style>
body, .stApp { background-color: #0f1115; color: #e0e6ed; }
/* KPI cards */
.block-container { padding-top: 1.4rem; }
[data-testid="stMetric"] { background: linear-gradient(135deg,#101824,#1d2b3a); padding: 0.85rem 1rem; border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,.4); }
[data-testid="stMetricLabel"] { color: #c3d3e8; font-weight: 600; }
[data-testid="stMetricValue"] { color: #f0f6ff; }
/* Dataframe tweaks */
[data-testid="stDataFrame"] div .row_heading, [data-testid="stDataFrame"] div .blank { background: #17212b !important; }
/* Expander style */
div.st-expander { border: 1px solid #1f3447; border-radius: 10px; }
/* Warning icon alignment */
.symbol-flag { font-size: .9rem; margin-left: .35rem; }
/* Buttons */
.stButton>button { background: linear-gradient(90deg,#334155,#1e293b); color:#e0e6ed; border:1px solid #334155; }
.stButton>button:hover { border-color:#64748b; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------- Auth (simple password gate) ----------------
import os as _os

# Enhanced password collection to avoid errors when secrets.toml is absent and to
# ensure gating whenever any password is configured via environment or secrets.

def _collect_allowed_passwords():
    allowed = set()
    # Environment variables (support single or comma-separated list)
    for key in ('APP_PASSWORD', 'APP_PASSWORDS'):
        val = _os.getenv(key, '')
        if val:
            allowed.update(p.strip() for p in val.split(',') if p.strip())
    # Secrets (graceful handling if secrets not defined or key missing)
    try:
        sec = getattr(st, 'secrets', {})
        for key in ('APP_PASSWORD', 'APP_PASSWORDS'):
            if isinstance(sec, dict) and key in sec:
                sval = str(sec[key])
                if sval:
                    allowed.update(p.strip() for p in sval.split(',') if p.strip())
            else:
                # st.secrets may be a config object supporting get
                try:
                    sval = sec.get(key) if hasattr(sec, 'get') else None
                except Exception:
                    sval = None
                if sval:
                    allowed.update(p.strip() for p in str(sval).split(',') if p.strip())
    except Exception:
        # Ignore any secrets access issues to avoid breaking the app
        pass
    return allowed

def _authenticate():
    allowed = _collect_allowed_passwords()
    if not allowed:  # No passwords configured -> open access
        return True
    if st.session_state.get('auth_ok'):
        return True
    st.title('🔒 Secure Dashboard')
    with st.form('login_form'):
        pw = st.text_input('Password', type='password')
        submit = st.form_submit_button('Enter')
        if submit:
            if pw in allowed:
                st.session_state['auth_ok'] = True
                st.success('Access granted')
                return True
            else:
                st.error('Invalid password')
    st.stop()

_authenticate()

@st.cache_data(ttl=300)
def load_data():
    raw = load_raw_sheet()
    holdings_summary = extract_holdings_summary(raw)
    tx = extract_transactions(raw)
    cost_basis = compute_cost_basis_positions(tx)
    reconciled = reconcile_positions_with_authoritative(cost_basis, holdings_summary)
    recon_stats = compute_reconciliation_stats(tx, cost_basis)
    flows = portfolio_invested_timeseries(tx)
    in_tx_not_summary, in_summary_not_tx = compute_discrepancy_sets(cost_basis, holdings_summary, tx)
    return tx, reconciled, holdings_summary, cost_basis, recon_stats, flows, (in_tx_not_summary, in_summary_not_tx)

# ---------------- Helpers ----------------

def fetch_prices_safe(symbols, vs):
    try:
        return fetch_prices(symbols, vs_currency=vs)
    except Exception as e:
        st.warning(f"Price fetch failed: {e}")
        return {}

def enrich_with_live_prices(positions: pd.DataFrame, base_fiat: str):
    symbols = positions['Crypto'].dropna().str.upper().unique().tolist()
    live = fetch_prices_safe(symbols, base_fiat)
    positions = positions.copy()
    positions['Crypto'] = positions['Crypto'].str.upper()
    positions['LivePrice'] = positions['Crypto'].map(live)
    positions['LiveValue'] = positions['Quantity'] * positions['LivePrice']
    positions['UnrealizedP&L'] = positions['LiveValue'] - positions['Invested']
    positions.loc[positions['Invested'].isna(), 'UnrealizedP&L'] = pd.NA
    positions['UnrealizedP&L%'] = (positions['UnrealizedP&L'] / positions['Invested'] * 100)
    positions.loc[positions['Invested'].isna(), 'UnrealizedP&L%'] = pd.NA
    return positions

def fmt_money(v, prec_small=6):
    if pd.isna(v):
        return '—'
    if v == 0:
        return '0'
    if abs(v) < 0.01:
        return f"{v:,.{prec_small}f}"
    return f"{v:,.2f}"

def fmt_pct(v):
    return '—' if pd.isna(v) else f"{v:,.2f}%"

# ---------------- Sidebar ----------------
base_fiat = os.getenv('BASE_FIAT','gbp').lower()
refresh_seconds = int(os.getenv('REFRESH_SECONDS','180'))

st.sidebar.markdown('### Settings')
base_fiat = st.sidebar.selectbox('Base Currency', ['gbp','usd','eur'], index=['gbp','usd','eur'].index(base_fiat))
refresh_seconds = st.sidebar.slider('Auto Refresh (s)', 0, 600, refresh_seconds, 30)
manual_refresh = st.sidebar.button('Refresh Now')
show_raw_timeseries = st.sidebar.checkbox('Show raw flow data table', False)
compact_mode = st.sidebar.checkbox('Mobile / Compact layout', False)
flow_metrics_selected = st.sidebar.multiselect(
    'Flow metrics to plot',
    ['GrossCumulativeBuys','NetCumulativeFlows','PortfolioCostBasis'],
    default=['GrossCumulativeBuys','NetCumulativeFlows','PortfolioCostBasis']
)
# Inject extra responsive CSS
st.markdown(
    """
    <style>
    @media (max-width: 820px) {
        .block-container { padding-left: .6rem !important; padding-right: .6rem !important; }
        [data-testid="stMetric"] { margin-bottom: .6rem; }
        .st-emotion-cache-1wmy9hl, .st-emotion-cache-13ln4jf { padding-top: .4rem !important; }
        .stPlotlyChart { height: 320px !important; }
    }
    </style>
    """,
    unsafe_allow_html=True
)
if manual_refresh:
    st.cache_data.clear()

with st.spinner('Loading & computing...'):
    tx, positions, holdings_summary, cost_basis, recon_stats, flows, discrepancy_sets = load_data()

# Discrepancy sets using simple aggregation baseline
raw_calc_positions = current_positions(tx)
in_tx_not_summary, in_summary_not_tx = compute_discrepancy_sets(raw_calc_positions, holdings_summary, tx)

positions_live = enrich_with_live_prices(positions, base_fiat)
missing = positions_live[positions_live['LivePrice'].isna()]['Crypto'].tolist()
if missing:
    st.warning(f"No live price for: {', '.join(missing)} (add override if needed)")
positions_live['Crypto'] = positions_live['Crypto'].str.upper()

# ---------------- Header ----------------
st.title('💹 Crypto Portfolio Dashboard')
st.caption(f"Updated {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC | Base: {base_fiat.upper()}")
if not holdings_summary.empty:
    st.info('Using sheet Holding quantities (authoritative). Average cost basis & realized P&L derived from transactions.')

# ---------------- KPIs ----------------
if not compact_mode:
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    kpi_targets = [col1, col2, col3, col4, col5, col6]
else:
    # Stack KPIs vertically for mobile
    kpi_targets = [st.container() for _ in range(6)]
portfolio_invested = positions_live['Invested'].fillna(0).sum()
portfolio_live = positions_live['LiveValue'].sum()
realized_total = positions_live['RealizedPnl'].fillna(0).sum()
unrealized_total = positions_live['UnrealizedP&L'].dropna().sum()
total_pl = realized_total + unrealized_total
pl_pct = total_pl / portfolio_invested * 100 if portfolio_invested else math.nan
net_flows = recon_stats['NetFlows']
with kpi_targets[0]:
    col_label = 'Active Cost Basis'
    st.metric(col_label, fmt_money(portfolio_invested) + f' {base_fiat.upper()}')
with kpi_targets[1]:
    st.metric('Current Value', fmt_money(portfolio_live) + f' {base_fiat.upper()}')
with kpi_targets[2]:
    st.metric('Realized P&L', fmt_money(realized_total) + f' {base_fiat.upper()}')
with kpi_targets[3]:
    st.metric('Unrealized P&L', fmt_money(unrealized_total) + f' {base_fiat.upper()}', fmt_pct(unrealized_total/portfolio_invested*100 if portfolio_invested else math.nan))
with kpi_targets[4]:
    st.metric('Total P&L', fmt_money(total_pl) + f' {base_fiat.upper()}', fmt_pct(pl_pct))
with kpi_targets[5]:
    st.metric('Net Cash Flows', fmt_money(net_flows) + f' {base_fiat.upper()}')

# ---------------- Allocation & P&L visuals ----------------
alloc_col, pnl_col = st.columns([1,1])
with alloc_col:
    alloc_fig = px.pie(positions_live.dropna(subset=['LiveValue']), names='Crypto', values='LiveValue', title='Allocation (Live Value)', hole=0.55)
    alloc_fig.update_traces(textposition='inside', textinfo='percent+label', hovertemplate='%{label}<br>%{percent} (%{value:,.2f})')
    alloc_fig.update_layout(height=320 if compact_mode else 400)
    st.plotly_chart(alloc_fig, use_container_width=True)
with pnl_col:
    pl_fig = px.bar(positions_live.sort_values('UnrealizedP&L', ascending=False), x='Crypto', y='UnrealizedP&L', title='Unrealized P&L by Asset', color='UnrealizedP&L', color_continuous_scale='RdYlGn')
    pl_fig.update_layout(coloraxis_showscale=False, height=320 if compact_mode else 400, margin=dict(t=60 if not compact_mode else 40), showlegend=False)
    st.plotly_chart(pl_fig, use_container_width=True)

# ---------------- Positions table ----------------
st.subheader('Positions Detail')
show_cols = ['Crypto','Quantity','QuantityTx','QuantityDiff','AvgCost','Invested','LivePrice','LiveValue','UnrealizedP&L','UnrealizedP&L%','RealizedPnl','MissingCost','NegativeQty']
fmt_df = positions_live[show_cols].copy()
for c in ['AvgCost','Invested','LivePrice','LiveValue','UnrealizedP&L','RealizedPnl']:
    if c in fmt_df.columns:
        fmt_df[c] = fmt_df[c].apply(lambda v: fmt_money(v))
if 'UnrealizedP&L%' in fmt_df:
    fmt_df['UnrealizedP&L%'] = fmt_df['UnrealizedP&L%'].apply(lambda v: fmt_pct(v))
fmt_df['MissingCost'] = fmt_df['MissingCost'].map({True:'⚠️', False:''})
fmt_df['NegativeQty'] = fmt_df['NegativeQty'].map({True:'⚠️', False:''})
# Reduce table height if compact
st.dataframe(fmt_df.sort_values('LiveValue', ascending=False), use_container_width=True, height=340 if compact_mode else 420)

# ---------------- Capital flows & cost basis timeseries ----------------
st.subheader('Capital Flows & Cost Basis')
if not tx.empty:
    flow_ts = portfolio_invested_timeseries(tx)
    if not flow_ts.empty:
        focus_cols = [c for c in flow_metrics_selected if c in flow_ts.columns]
        if not focus_cols:
            focus_cols = ['PortfolioCostBasis']
        multi_long = flow_ts.melt(id_vars='Date', value_vars=focus_cols, var_name='Metric', value_name='Amount')
        line_multi = px.line(multi_long, x='Date', y='Amount', color='Metric', title='Selected Flow Metrics Over Time')
        line_multi.update_yaxes(title=f'Amount ({base_fiat.upper()})')
        line_multi.update_layout(height=340 if compact_mode else 480)
        st.plotly_chart(line_multi, use_container_width=True)
        # Add explanatory callout
        with st.expander('How to interpret these flow metrics'):
            st.markdown(
                '• GrossCumulativeBuys: Sum of every buy Value ever (can exceed active capital).\n'
                '• NetCumulativeFlows: Gross buys minus sell proceeds (cash still deployed net of withdrawals).\n'
                '• PortfolioCostBasis: Cost basis of current holdings only (excludes disposed positions).\n'
                'Large gap GrossCumulativeBuys - PortfolioCostBasis = capital recycled via realized trades.'
            )
        if show_raw_timeseries:
            st.dataframe(flow_ts, use_container_width=True, height=300)
    else:
        st.info('No flow data computed.')
else:
    st.info('No transactions parsed.')

# ---------------- Reconciliation panel ----------------
with st.expander('Reconciliation & Integrity Checks', expanded=False):
    st.markdown('**High-level cash & P&L reconciliation (average cost method)**')
    rcols = st.columns(4)
    rcols[0].metric('Gross Buys', fmt_money(recon_stats['GrossBuys']))
    rcols[1].metric('Sell Proceeds', fmt_money(recon_stats['Proceeds']))
    rcols[2].metric('Net Flows', fmt_money(recon_stats['NetFlows']))
    rcols[3].metric('Cost Basis (Active)', fmt_money(recon_stats['PortfolioCostBasis']))
    rcols2 = st.columns(4)
    rcols2[0].metric('Realized P&L', fmt_money(recon_stats['RealizedPnlTotal']))
    rcols2[1].metric('Implied Cost of Units Sold', fmt_money(recon_stats['ImpliedCostUnitsSold']))
    rcols2[2].metric('Realized Check (Proceeds - Implied Cost)', fmt_money(recon_stats['RealizedCheck']))
    diff_display = fmt_money(recon_stats['RealizedDiff'])
    rcols2[3].metric('Realized Diff', diff_display)
    if abs(recon_stats['RealizedDiff']) > 1e-6:
        st.warning('Realized P&L mismatch beyond rounding. Review transactions for sign/duplication issues.')
    if in_summary_not_tx:
        st.write('Summary-only symbols:', ', '.join(in_summary_not_tx))
    if in_tx_not_summary:
        st.write('Tx-only symbols (excluded):', ', '.join(in_tx_not_summary))

st.caption('Figures use average cost method. Validate before decisions. Live prices via CoinGecko.')

# ---------------- Auto refresh ----------------
if refresh_seconds:
    st.markdown(f'<meta http-equiv="refresh" content="{refresh_seconds}">', unsafe_allow_html=True)

# -------------- Optional Snapshot & Export -------------- #
with st.expander('Snapshot & Export'):
    prefix = st.text_input('Optional snapshot prefix', value='manual')
    if st.button('Persist Snapshot to /snapshots folder'):
        try:
            folder = persist_portfolio_snapshot(tx, cost_basis, positions, flows, prefix=prefix)
            st.success(f'Snapshot saved to {folder}')
        except Exception as e:
            st.error(f'Failed to persist snapshot: {e}')
    csv_choice = st.selectbox('Export DataFrame to CSV', ['None','Reconciled Positions','Ledger','Transactions','Timeseries'])
    if csv_choice != 'None':
        if csv_choice == 'Reconciled Positions':
            df_dl = positions_live[show_cols]
        elif csv_choice == 'Ledger':
            df_dl = average_cost_ledger(tx)
        elif csv_choice == 'Transactions':
            df_dl = tx
        else:
            df_dl = flows
        st.download_button('Download CSV', df_dl.to_csv(index=False), file_name=f'{csv_choice.lower().replace(" ","_")}.csv', mime='text/csv')

# -------------- Footer -------------- #
st.caption('Dust threshold applied: values below {:.2e} treated as zero for reconciliation diffs.'.format(DUST_THRESHOLD))
