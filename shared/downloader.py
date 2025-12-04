import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

def fetch_for_ticker(pro, ts_code: str, start_date: str, fields_daily: str, fields_daily_basic: str, fields_moneyflow: str, timeout_moneyflow=10):
    """Fetch daily, daily_basic, moneyflow for a single ts_code in parallel.

    Returns a dict with keys 'daily','daily_basic','moneyflow' mapped to DataFrames (or empty DataFrame on failure).
    """
    results: Dict[str, object] = {}

    def f_daily():
        return pro.daily(ts_code=ts_code, start_date=start_date, end_date="", fields=fields_daily)

    def f_daily_basic():
        return pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date="", fields=fields_daily_basic)

    def f_moneyflow():
        return pro.moneyflow(ts_code=ts_code, start_date=start_date, end_date="", fields=fields_moneyflow)

    # Use a small threadpool to run the three requests in parallel for this ticker
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {
            ex.submit(f_daily): 'daily',
            ex.submit(f_daily_basic): 'daily_basic',
            ex.submit(f_moneyflow): 'moneyflow'
        }

        for fut in as_completed(futs):
            name = futs[fut]
            try:
                if name == 'moneyflow':
                    # Respect moneyflow timeout by waiting with timeout
                    # Note: as_completed doesn't support per-future timeout easily, but we guard by try/except
                    val = fut.result(timeout=timeout_moneyflow)
                else:
                    val = fut.result()
                results[name] = val
            except Exception:
                # return empty DataFrame on any failure for graceful degradation
                try:
                    import pandas as pd
                    results[name] = pd.DataFrame()
                except Exception:
                    results[name] = None

    # Ensure keys exist
    for k in ('daily','daily_basic','moneyflow'):
        if k not in results:
            try:
                import pandas as pd
                results[k] = pd.DataFrame()
            except Exception:
                results[k] = None

    return results


def fetch_batch(pro, ts_codes, start_date: str, fields_daily: str, fields_daily_basic: str, fields_moneyflow: str, timeout=30, max_codes_per_request: int = 10):
    """Fetch multiple ts_codes in (safe) sub-requests and merge results.

    Some servers limit the number of ts_code values accepted in a single request.
    To be robust we split `ts_codes` into chunks of at most `max_codes_per_request`.

    Returns a dict: {'daily': {ts: df,...}, 'daily_basic': {...}, 'moneyflow': {...}}
    """
    import pandas as pd
    if not ts_codes:
        return {'daily': {}, 'daily_basic': {}, 'moneyflow': {}}

    results = {'daily': {}, 'daily_basic': {}, 'moneyflow': {}}

    # helper to split df by ts_code
    def split_df_by_ts(df):
        out = {}
        if df is None or df.empty:
            return out
        if 'ts_code' not in df.columns:
            out['_all'] = df
            return out
        for code, g in df.groupby('ts_code'):
            out[code] = g.reset_index(drop=True)
        return out

    # process in safe-sized chunks
    for i in range(0, len(ts_codes), max_codes_per_request):
        chunk = ts_codes[i:i+max_codes_per_request]
        ts_param = ','.join(chunk)

        # daily
        try:
            df_daily = pro.daily(ts_code=ts_param, start_date=start_date, end_date="")
            if df_daily is None:
                df_daily = pd.DataFrame()
        except Exception:
            df_daily = pd.DataFrame()

        # daily_basic
        try:
            df_basic = pro.daily_basic(ts_code=ts_param, start_date=start_date, end_date="")
            if df_basic is None:
                df_basic = pd.DataFrame()
        except Exception:
            df_basic = pd.DataFrame()

        # moneyflow (with timeout wrapper)
        try:
            from concurrent.futures import ThreadPoolExecutor
            def call_mf():
                return pro.moneyflow(ts_code=ts_param, start_date=start_date, end_date="")
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(call_mf)
                try:
                    df_flow = fut.result(timeout=timeout)
                    if df_flow is None:
                        df_flow = pd.DataFrame()
                except Exception:
                    fut.cancel()
                    df_flow = pd.DataFrame()
        except Exception:
            df_flow = pd.DataFrame()

        # merge results for this chunk
        daily_map = split_df_by_ts(df_daily)
        basic_map = split_df_by_ts(df_basic)
        flow_map = split_df_by_ts(df_flow)

        # update global results
        results['daily'].update(daily_map)
        results['daily_basic'].update(basic_map)
        results['moneyflow'].update(flow_map)

    return results
