"""Insert one experiment row into the leaderboard feature group via the SDK.

The `hops fg insert --file` CLI cannot produce a timestamp column from JSON (it
derives string/bigint and the schema rejects it), so the ts column goes through
the SDK with a real pandas datetime.

usage: log_row.py <commit> <val_metric> <peak_gb> <status> <description> <iso_ts>
"""
import sys

import pandas as pd
import hopsworks

TAG = "astjun29"


def main():
    commit, val, mem, status, desc, ts = sys.argv[1:7]
    fs = hopsworks.login().get_feature_store()
    fg = fs.get_feature_group(f"autoresearch_experiments_{TAG}", version=1)
    fg.insert(pd.DataFrame([{
        "commit": commit,
        "val_metric": float(val),
        "peak_memory_gb": float(mem),
        "status": status,
        "description": desc,
        "ts": pd.to_datetime(ts, utc=True),
    }]), wait=True)
    print(f"inserted {commit} {status} {val}")


if __name__ == "__main__":
    main()
