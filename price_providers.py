import os
import time
import requests
from functools import lru_cache
from typing import List, Dict, Tuple

COINGECKO_API_BASE = os.getenv('COINGECKO_API_BASE','https://api.coingecko.com/api/v3')
API_KEY = os.getenv('COINGECKO_API_KEY')
HEADERS = {'accept':'application/json'}
if API_KEY:
    HEADERS['x-cg-pro-api-key'] = API_KEY

# Manual overrides for ambiguous / important symbols (legacy symbols map to current project id)
OVERRIDE_IDS: Dict[str,str] = {
    'btc':'bitcoin',
    'eth':'ethereum',
    'egld':'multiversx',      # EGLD (MultiversX)
    'erd':'multiversx',       # Legacy ERD -> EGLD
    'mex':'xexchange-token',
    'ride':'holoride',
    'near':'near',
    'link':'chainlink',
    'flow':'flow',
    'dia':'dia',
    'grt':'the-graph',
    'usdt':'tether',
    'sol':'solana',
    'uni':'uniswap',
    'eos':'eos',
    'xrp':'ripple'
}

SYMBOL_MAPPING_CACHE: Dict[str,str] = {}
SYMBOL_CANDIDATES: Dict[str, List[str]] = {}

@lru_cache(maxsize=1)
def _coin_list():
    url = f"{COINGECKO_API_BASE}/coins/list"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not SYMBOL_CANDIDATES:
        for c in data:
            sym = c['symbol'].lower().strip()
            SYMBOL_CANDIDATES.setdefault(sym, []).append(c['id'])
    return data


def symbol_to_id(symbol: str) -> str:
    s = symbol.lower().strip()
    if s in SYMBOL_MAPPING_CACHE:
        return SYMBOL_MAPPING_CACHE[s]
    if s in OVERRIDE_IDS:  # override first
        SYMBOL_MAPPING_CACHE[s] = OVERRIDE_IDS[s]
        return OVERRIDE_IDS[s]
    _coin_list()  # populate candidates
    candidates = SYMBOL_CANDIDATES.get(s, [])
    if not candidates:
        raise KeyError(f'No CoinGecko id found for symbol {symbol}')
    if len(candidates) > 1:
        # Heuristics: prefer exact symbol id, then shortest id containing the symbol, else first
        exact = [c for c in candidates if c == s]
        if exact:
            chosen = exact[0]
        else:
            containing = [c for c in candidates if s in c]
            chosen = sorted(containing, key=len)[0] if containing else sorted(candidates, key=len)[0]
    else:
        chosen = candidates[0]
    SYMBOL_MAPPING_CACHE[s] = chosen
    return chosen


def _do_simple_price(ids: List[str], vs_currency: str) -> Dict[str, Dict[str, float]]:
    url = f"{COINGECKO_API_BASE}/simple/price"
    params = {'ids':','.join(sorted(set(ids))), 'vs_currencies':vs_currency}
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_prices_detailed(symbols: List[str], vs_currency: str = 'gbp') -> Tuple[Dict[str,float], List[str], Dict[str,str]]:
    """Return (price_map, unresolved_symbols, sym_to_id) for diagnostics."""
    cleaned = []
    for s in symbols:
        if isinstance(s, str) and s.strip():
            cleaned.append(s.strip())
    sym_to_id: Dict[str,str] = {}
    unresolved: List[str] = []
    for sym in cleaned:
        try:
            sym_to_id[sym.upper()] = symbol_to_id(sym)
        except KeyError:
            unresolved.append(sym.upper())
    if not sym_to_id:
        return {}, unresolved, sym_to_id
    try:
        data = _do_simple_price(list(sym_to_id.values()), vs_currency)
    except Exception:
        return {}, cleaned, sym_to_id  # total failure
    price_map: Dict[str,float] = {}
    for disp_sym, cid in sym_to_id.items():
        try:
            price_map[disp_sym] = data[cid][vs_currency]
        except Exception:
            unresolved.append(disp_sym)
    return price_map, sorted(set(unresolved)), sym_to_id


def fetch_prices(symbols: List[str], vs_currency: str = 'gbp') -> Dict[str,float]:
    price_map, _, _ = fetch_prices_detailed(symbols, vs_currency)
    return price_map
