import pandas as pd
from pathlib import Path
import os
from datetime import datetime
from typing import Union, Optional

PORTFOLIO_FILE = Path(__file__).parent / 'Crypto Portfolio.xlsx'

HEADER_ROW_MARKER = 'Date'
CORE_COLUMNS = ['Date','Crypto','Quantity','Price','Value','P&L','Breakeven','Current Value','£','%']

# Symbol merge (legacy -> current)
SYMBOL_MERGE = { 'ERD':'EGLD' }

# Tolerance for treating tiny residual quantities/differences as zero ("dust")
DUST_THRESHOLD = 1e-8


def normalize_symbol(s):
    if pd.isna(s):
        return s
    return str(s).strip().upper()


def load_raw_sheet():
    df = pd.read_excel(PORTFOLIO_FILE, sheet_name='Crypto', header=0)
    return df


def extract_holdings_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the pre-table summary that contains authoritative Holding quantities.
    Looks at all rows before the header row marker and returns columns Crypto, Holding, Current Value if present.
    """
    # Locate header index as in extract_transactions
    header_idx = df.index[df.iloc[:,0].astype(str).str.strip().str.lower() == HEADER_ROW_MARKER.lower()]
    if len(header_idx)==0:
        return pd.DataFrame(columns=['Crypto','Holding','Current Value'])
    h = header_idx[0]
    pre = df.iloc[:h].copy()
    # Keep rows with a crypto symbol and a numeric Holding
    if 'Crypto' not in pre.columns or 'Holding' not in pre.columns:
        return pd.DataFrame(columns=['Crypto','Holding','Current Value'])
    pre = pre[pre['Crypto'].notna()].copy()
    pre['Crypto'] = pre['Crypto'].apply(normalize_symbol)
    pre['Holding'] = pd.to_numeric(pre.get('Holding'), errors='coerce')
    pre['Current Value'] = pd.to_numeric(pre.get('Current Value'), errors='coerce')
    pre = pre[pre['Holding'].notna()]
    return pre[['Crypto','Holding','Current Value']]


def extract_transactions(df: pd.DataFrame) -> pd.DataFrame:
    # Find the header row where first column == 'Date'
    header_idx = df.index[df.iloc[:,0].astype(str).str.strip().str.lower() == HEADER_ROW_MARKER.lower()]
    if len(header_idx)==0:
        raise ValueError('Could not locate header row with marker Date')
    h = header_idx[0]
    header = df.iloc[h].tolist()
    data = df.iloc[h+1:].reset_index(drop=True)
    data.columns = header + data.columns[len(header):].tolist()
    # Preserve original order index for deterministic processing
    data['__ROW_ORDER'] = range(len(data))
    # Keep only rows that have a Date value
    data = data[~data['Date'].isna()].copy()
    data.rename(columns=lambda c: str(c).strip(), inplace=True)
    if 'Crypto' in data.columns:
        data['Crypto'] = data['Crypto'].ffill().apply(normalize_symbol)
        data['Crypto'] = data['Crypto'].replace(SYMBOL_MERGE)
    for col in ['Quantity','Price','Value','P&L','Breakeven','Current Value','£','%']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce', dayfirst=True)
    data = data[(data['Crypto'].notna()) & (data['Quantity'].notna())]
    # Sign normalization:
    # Represent buys: Quantity>0, Value>0 ; sells: Quantity>0, Value<0. Adjust inconsistent rows.
    def _norm_row(r):
        q = r['Quantity']
        v = r['Value']
        if pd.isna(q) or pd.isna(v):
            return q, v
        if q < 0 and v > 0:
            # Negative qty but positive cash outflow -> treat as sell (flip value sign and quantity sign)
            return -q, -v
        if q < 0 and v < 0:
            # Both negative -> sell; make quantity positive
            return -q, v
        if q > 0 and v > 0:
            # Buy ok
            return q, v
        if q > 0 and v < 0:
            # Sell already in canonical form
            return q, v
        return q, v
    adj = data.apply(lambda r: _norm_row(r), axis=1, result_type='expand')
    data['Quantity'], data['Value'] = adj[0], adj[1]
    # Filter any zero or NaN quantity
    data = data[data['Quantity']>0]
    cols = ['Date','Crypto','Quantity','Price','Value']
    opt_cols = [c for c in ['P&L','Breakeven','Current Value'] if c in data.columns]
    cols += opt_cols + ['__ROW_ORDER']
    return data[cols]


def current_positions(df: pd.DataFrame) -> pd.DataFrame:
    # Aggregate by Crypto symbol summing quantity and separating buy cost vs sells
    def invested(series):
        # Sum only positive Value entries (buys) as cost basis
        return series[series > 0].sum()
    grp = df.groupby('Crypto', dropna=True).agg(
        Quantity=('Quantity','sum'),
        Invested=('Value', invested),
        NetFlows=('Value','sum'),  # buys - sells
        GrossBuys=('Value', lambda s: s[s>0].sum()),
        GrossSells=('Value', lambda s: -s[s<0].sum())
    ).reset_index()
    return grp

# ---------------- New cost basis / realized P&L logic ---------------- #

def _average_cost_ledger(transactions: pd.DataFrame) -> pd.DataFrame:
    # Rewritten to rely on normalized signs (Quantity>0 always)
    if transactions.empty:
        return pd.DataFrame(columns=['Date','Crypto','Quantity','Value','RunQty','AvgCost','RealizedPnlTrade','CumRealizedPnl'])
    tx = transactions.copy().sort_values(['Crypto','Date','__ROW_ORDER']).reset_index(drop=True)
    rows = []
    for sym, sub in tx.groupby('Crypto', sort=False):
        run_qty = 0.0
        avg_cost = 0.0
        cum_realized = 0.0
        for _, r in sub.iterrows():
            qty = float(r['Quantity'])
            val = float(r['Value'])
            realized_trade = 0.0
            if val > 0:  # buy
                new_qty = run_qty + qty
                if new_qty > 0:
                    avg_cost = (run_qty * avg_cost + val) / new_qty
                run_qty = new_qty
            elif val < 0:  # sell
                sell_qty = qty
                if sell_qty > run_qty:  # oversell -> limit to available for P&L
                    effective = run_qty
                else:
                    effective = sell_qty
                proceeds = -val
                realized_trade = proceeds - effective * avg_cost
                cum_realized += realized_trade
                run_qty -= sell_qty
                if run_qty < 0:
                    run_qty = 0  # do not allow negative inventory after normalization
            rows.append({
                'Date': r['Date'],
                'Crypto': sym,
                'Quantity': qty,
                'Value': val,
                'RunQty': run_qty,
                'AvgCost': avg_cost,
                'RealizedPnlTrade': realized_trade,
                'CumRealizedPnl': cum_realized
            })
    return pd.DataFrame(rows)


def compute_cost_basis_positions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Compute per-asset cost basis & realized P&L using average cost method.
    Returns columns:
      Crypto, QuantityTx (ending), AvgCost, InvestedTx (ending quantity * avg cost), RealizedPnl
    """
    ledger = _average_cost_ledger(transactions)
    if ledger.empty:
        return pd.DataFrame(columns=['Crypto','QuantityTx','AvgCost','InvestedTx','RealizedPnl'])
    last = ledger.sort_values('Date').groupby('Crypto').tail(1)
    out = last[['Crypto','RunQty','AvgCost','CumRealizedPnl']].rename(columns={
        'RunQty':'QuantityTx',
        'CumRealizedPnl':'RealizedPnl'
    }).reset_index(drop=True)
    out['InvestedTx'] = out['QuantityTx'] * out['AvgCost']
    return out[['Crypto','QuantityTx','AvgCost','InvestedTx','RealizedPnl']]


def apply_authoritative_holdings(positions: pd.DataFrame, holdings_summary: pd.DataFrame) -> pd.DataFrame:
    """Return only symbols present in the holdings summary, using its Holding quantity.
    Transaction aggregates (Invested etc.) are merged when available.
    Any symbol in summary but missing transactions will have NaN Invested.
    """
    if holdings_summary.empty:
        return positions
    hs = holdings_summary.copy()
    hs['Crypto'] = hs['Crypto'].apply(normalize_symbol)
    pos = positions.copy()
    pos['Crypto'] = pos['Crypto'].apply(normalize_symbol)
    merged = hs.merge(pos, on='Crypto', how='left')
    merged.rename(columns={'Holding':'QuantitySummary'}, inplace=True)
    merged['Quantity'] = merged['QuantitySummary']
    merged.drop(columns=['QuantitySummary'], inplace=True)
    # Drop any rows where Quantity <= 0 just in case
    merged = merged[merged['Quantity'] > 0].reset_index(drop=True)
    return merged


def reconcile_positions_with_authoritative(cost_basis_positions: pd.DataFrame, holdings_summary: pd.DataFrame, dust_threshold: float = DUST_THRESHOLD) -> pd.DataFrame:
    """Combine cost basis (from transactions) with authoritative holdings summary.
    Adds:
      Quantity (authoritative), QuantityTx, QuantityDiff,
      Invested (scaled from AvgCost * authoritative quantity),
      Flags for MissingCost (authoritative quantity but no tx data) & NegativeQty (tx derived negative ending qty)
    Applies dust threshold to clamp negligible diffs/quantities.
    """
    if holdings_summary.empty:
        cb = cost_basis_positions.copy()
        cb['Quantity'] = cb['QuantityTx']
        cb['QuantityDiff'] = 0.0
        cb['Invested'] = cb['InvestedTx']
        cb['MissingCost'] = False
        cb['NegativeQty'] = cb['QuantityTx'] < 0
        # Dust clamp
        cb.loc[cb['Quantity'].abs() < dust_threshold, 'Quantity'] = 0.0
        cb.loc[cb['QuantityTx'].abs() < dust_threshold, 'QuantityTx'] = 0.0
        cb['QuantityDiff'] = cb['Quantity'] - cb['QuantityTx']
        cb.loc[cb['QuantityDiff'].abs() < dust_threshold, 'QuantityDiff'] = 0.0
        cb['NegativeQty'] = (cb['QuantityTx'] < 0) & (cb['QuantityTx'].abs() >= dust_threshold)
        return cb
    hs = holdings_summary.copy()
    hs['Crypto'] = hs['Crypto'].apply(normalize_symbol)
    cb = cost_basis_positions.copy()
    cb['Crypto'] = cb['Crypto'].apply(normalize_symbol)
    merged = hs.merge(cb, on='Crypto', how='left')
    merged.rename(columns={'Holding':'Quantity'}, inplace=True)
    merged['QuantityTx'] = merged['QuantityTx'].fillna(0)
    merged['QuantityDiff'] = merged['Quantity'] - merged['QuantityTx']
    # Dust clamping
    for col in ['Quantity','QuantityTx','QuantityDiff']:
        merged.loc[merged[col].abs() < dust_threshold, col] = 0.0
    # AvgCost rules: if tx data missing, leave AvgCost NaN
    merged['Invested'] = merged.apply(
        lambda r: r['AvgCost'] * r['Quantity'] if pd.notna(r['AvgCost']) and r['Quantity']>0 else pd.NA, axis=1
    )
    merged['MissingCost'] = merged['AvgCost'].isna()
    merged['NegativeQty'] = (merged['QuantityTx'] < 0) & (merged['QuantityTx'].abs() >= dust_threshold)
    return merged[['Crypto','Quantity','QuantityTx','QuantityDiff','AvgCost','Invested','InvestedTx','RealizedPnl','MissingCost','NegativeQty']]


def compute_discrepancy_sets(positions: pd.DataFrame, holdings_summary: pd.DataFrame, transactions: pd.DataFrame):
    tx_syms = set(transactions['Crypto'].unique())
    summary_syms = set(holdings_summary['Crypto'].unique())
    in_tx_not_summary = sorted(tx_syms - summary_syms)
    in_summary_not_tx = sorted(summary_syms - tx_syms)
    return in_tx_not_summary, in_summary_not_tx

def average_cost_ledger(transactions: pd.DataFrame) -> pd.DataFrame:
    """Public wrapper returning the average cost ledger per transaction (see _average_cost_ledger)."""
    return _average_cost_ledger(transactions)


def portfolio_invested_timeseries(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return a chronological portfolio-level timeseries with:
      Date
      GrossCumulativeBuys  (sum of all positive Value so far)
      NetCumulativeFlows   (sum of Value, buys minus sells)
      PortfolioCostBasis   (sum over assets of RunQty * AvgCost after each txn)
      RealizedPnlToDate    (cumulative realized P&L across assets)
    This differentiates between *gross money ever put in* vs *current cost basis still deployed*.
    """
    if transactions.empty:
        return pd.DataFrame(columns=['Date','GrossCumulativeBuys','NetCumulativeFlows','PortfolioCostBasis','RealizedPnlToDate'])
    ledger = average_cost_ledger(transactions)
    # Per row cost basis for that asset after the transaction
    ledger['AssetCostBasis'] = ledger.apply(lambda r: max(r['RunQty'],0) * r['AvgCost'], axis=1)
    # Iterate chronologically, maintain latest asset basis & realized pnl
    latest_basis = {}
    latest_realized = {}
    rows = []
    gross_buys = 0.0
    net_flows = 0.0
    for _, row in ledger.sort_values('Date').iterrows():
        val = row['Value']
        if val > 0:
            gross_buys += val
        net_flows += val
        latest_basis[row['Crypto']] = row['AssetCostBasis']
        latest_realized[row['Crypto']] = row['CumRealizedPnl']
        portfolio_basis = sum(latest_basis.values())
        realized_total = sum(latest_realized.values())
        rows.append({
            'Date': row['Date'],
            'GrossCumulativeBuys': gross_buys,
            'NetCumulativeFlows': net_flows,
            'PortfolioCostBasis': portfolio_basis,
            'RealizedPnlToDate': realized_total
        })
    ts = pd.DataFrame(rows)
    # Collapse same day multiple transactions to last state
    ts = ts.sort_values('Date').groupby('Date').tail(1).reset_index(drop=True)
    return ts

def compute_reconciliation_stats(transactions: pd.DataFrame, cost_basis_positions: pd.DataFrame) -> dict:
    gross_buys = transactions.loc[transactions['Value']>0, 'Value'].sum()
    proceeds = -transactions.loc[transactions['Value']<0, 'Value'].sum()
    net_flows = transactions['Value'].sum()
    portfolio_cost_basis = cost_basis_positions['InvestedTx'].sum()
    realized_total = cost_basis_positions['RealizedPnl'].sum()
    # Implied cost of units sold under avg cost: gross_buys - remaining cost basis
    implied_cost_units_sold = gross_buys - portfolio_cost_basis
    realized_check = proceeds - implied_cost_units_sold
    return {
        'GrossBuys': gross_buys,
        'Proceeds': proceeds,
        'NetFlows': net_flows,
        'PortfolioCostBasis': portfolio_cost_basis,
        'RealizedPnlTotal': realized_total,
        'ImpliedCostUnitsSold': implied_cost_units_sold,
        'RealizedCheck': realized_check,
        'RealizedDiff': realized_total - realized_check
    }

# ---------------- Snapshot persistence ---------------- #

def persist_portfolio_snapshot(transactions: pd.DataFrame,
                               cost_basis_positions: pd.DataFrame,
                               reconciled_positions: pd.DataFrame,
                               timeseries: pd.DataFrame,
                               out_dir: Union[Path, str] = Path(__file__).parent / 'snapshots',
                               prefix: Optional[str] = None) -> Path:
    """Persist CSV snapshots (ledger-level, cost basis positions, reconciled positions, flow timeseries).
    Creates a dated subfolder to allow historical performance review independent of live session.
    Returns the folder path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    if prefix:
        folder = out_dir / f"{stamp}_{prefix}"
    else:
        folder = out_dir / stamp
    folder.mkdir(parents=True, exist_ok=True)
    # Save
    try:
        transactions.to_csv(folder / 'transactions.csv', index=False)
        average_cost_ledger(transactions).to_csv(folder / 'average_cost_ledger.csv', index=False)
        cost_basis_positions.to_csv(folder / 'cost_basis_positions.csv', index=False)
        reconciled_positions.to_csv(folder / 'reconciled_positions.csv', index=False)
        timeseries.to_csv(folder / 'portfolio_timeseries.csv', index=False)
    except Exception as e:
        # Best-effort; raise for visibility
        raise RuntimeError(f"Failed to persist snapshot: {e}")
    return folder

if __name__ == '__main__':
    raw = load_raw_sheet()
    hold = extract_holdings_summary(raw)
    tx = extract_transactions(raw)
    pos_calc = current_positions(tx)
    from pprint import pprint
    cost_basis = compute_cost_basis_positions(tx)
    stats = compute_reconciliation_stats(tx, cost_basis)
    reconciled = reconcile_positions_with_authoritative(cost_basis, hold)
    print('Holdings summary:\n', hold)
    print('\nTransactions normalized head:\n', tx.head())
    print('\nCost basis positions:\n', cost_basis)
    print('\nReconciled positions (authoritative quantities):')
    pprint(reconciled)
    print('\nReconciliation stats:')
    pprint(stats)
    print('\nFlow timeseries tail:')
    flows = portfolio_invested_timeseries(tx)
    print(flows.tail())
    print('\nTx-only symbols, Summary-only symbols:', compute_discrepancy_sets(pos_calc, hold, tx))
    # Persist snapshot demo
    try:
        snap_folder = persist_portfolio_snapshot(tx, cost_basis, reconciled, flows, prefix='manual_run')
        print(f"\nSnapshot saved to: {snap_folder}")
    except Exception as e:
        print(f"Snapshot persistence failed: {e}")
