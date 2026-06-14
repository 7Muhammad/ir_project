#!/usr/bin/env python3
"""
aggregate_rank_changes.py

Aggregates per-document rank-change files produced by evaluate.py into a
single long-format TSV that make_scale_table.py can consume directly.

Input directory structure:
    <rank_changes_dir>/
        dl19/   {token}_{mode}_{n}_bm25_19_{model}_rank_changes.tsv.gz
        dl20/   {token}_{mode}_{n}_bm25_20_{model}_rank_changes.tsv.gz

Output TSV columns (written to <out_file>):
    token, model, dataset, metric, value, position, n_tok, sig

    metric is one of: MRC | Success Rate | sig
    MRC          = mean rank change (positive = attack boosted rank)
    Success Rate = fraction of documents that moved up in rank
    sig          = 1 if the Wilcoxon test is significant, else 0

Selection logic:
    For each (token, model) pair, the (position, n_tok) combination with the
    highest mean MRC averaged across both datasets is selected as the best
    configuration. All rows for that (token, model) share the same position
    and n_tok — this matches the expectation of make_scale_table.py.

Significance:
    One-sample Wilcoxon signed-rank test on rank_change values (H0: median = 0).
    Bonferroni correction applied across all (token, model, dataset) tests.

USAGE:
    python aggregate_rank_changes.py \\
        --rank_changes_dir /path/to/runs/rank_changes \\
        --out_file aggregated.tsv

    # Then generate the LaTeX table:
    python ecir24-adversarial-evaluation/advseq2seq/stuffing/table_generation/make_scale_table.py \\
        --run_file aggregated.tsv --out_file table3.tex
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from fire import Fire

DATASET_MAP = {
    '19': 'dl19',
    '20': 'dl20',
}


def parse_filename(fname):
    """
    Parse {token}_{mode}_{n}_bm25_{year}_{model}_rank_changes.tsv.gz

    Splitting from the right is safe because:
      - model names (t5.small, t5.base, t5.large, t5.3b) contain dots, not underscores
      - token names (relevanttrue, informationbar, etc.) contain no underscores

    Returns: (token, position, n_tok, dataset, model)
    """
    base = fname.replace('_rank_changes.tsv.gz', '')
    parts = base.split('_')
    # parts layout: [token, mode, n, 'bm25', year, model]
    model   = parts[-1]           # e.g. 't5.base'
    year    = parts[-2].replace('.gz', '')  # e.g. '19.gz' → '19'
    # parts[-3] == 'bm25'
    n_tok   = int(parts[-4])      # e.g. 3
    mode    = parts[-5]           # e.g. 'start'
    token   = '_'.join(parts[:-5])  # e.g. 'relevanttrue'
    dataset = DATASET_MAP.get(year, f'dl{year}')
    return token, mode, n_tok, dataset, model


def main(rank_changes_dir: str, out_file: str, alpha: float = 0.05):
    stats = []   # one dict per file (scalar metrics)
    arrays = {}  # (token, model, dataset, mode, n_tok) -> np.array of rank_changes

    for dataset_folder in ['dl19', 'dl20']:
        folder = os.path.join(rank_changes_dir, dataset_folder)
        if not os.path.isdir(folder):
            print(f'WARNING: {folder} not found, skipping')
            continue

        fnames = sorted(f for f in os.listdir(folder) if f.endswith('_rank_changes.tsv.gz'))
        print(f'  {dataset_folder}: found {len(fnames)} rank_change files')

        for fname in fnames:
            try:
                token, mode, n_tok, dataset, model = parse_filename(fname)
            except Exception as e:
                print(f'  WARNING: could not parse "{fname}": {e}')
                continue

            df = pd.read_csv(os.path.join(folder, fname), sep='\t', index_col=False)
            mrc = float(df['rank_change'].mean())
            sr  = float(df['success'].mean())

            key = (token, model, dataset, mode, n_tok)
            arrays[key] = df['rank_change'].values
            stats.append({
                'token':    token,
                'model':    model,
                'dataset':  dataset,
                'position': mode,
                'n_tok':    n_tok,
                'mrc':      mrc,
                'sr':       sr,
            })

    full = pd.DataFrame(stats)
    if full.empty:
        raise RuntimeError(f'No rank_change files found under {rank_changes_dir}')

    print(f'\nModels found: {sorted(full.model.unique())}')
    print(f'Tokens found: {sorted(full.token.unique())}')

    # -------------------------------------------------------------------------
    # Select best (position, n_tok) per (token, model)
    # Criterion: highest mean MRC averaged across both datasets
    # -------------------------------------------------------------------------
    mean_mrc = (
        full.groupby(['token', 'model', 'position', 'n_tok'])['mrc']
        .mean()
        .reset_index()
        .rename(columns={'mrc': 'mean_mrc'})
    )
    best_idx = mean_mrc.groupby(['token', 'model'])['mean_mrc'].idxmax()
    best_configs = (
        mean_mrc.loc[best_idx, ['token', 'model', 'position', 'n_tok']]
        .reset_index(drop=True)
    )

    filtered = full.merge(best_configs, on=['token', 'model', 'position', 'n_tok'], how='inner')

    # -------------------------------------------------------------------------
    # Significance test: Wilcoxon signed-rank on rank_change, Bonferroni-corrected
    # -------------------------------------------------------------------------
    n_tests = len(filtered)
    corrected_alpha = alpha / n_tests

    out_rows = []
    for _, row in filtered.iterrows():
        key = (row['token'], row['model'], row['dataset'], row['position'], row['n_tok'])
        rank_changes = arrays[key]

        nonzero = rank_changes[rank_changes != 0]
        if len(nonzero) < 10:
            # Too few nonzero values for a reliable test
            p_val = 1.0
        else:
            try:
                _, p_val = wilcoxon(rank_changes)
            except Exception:
                p_val = 1.0

        sig = 1 if p_val < corrected_alpha else 0

        base = {
            'token':    row['token'],
            'model':    row['model'],
            'dataset':  row['dataset'],
            'position': row['position'],
            'n_tok':    int(row['n_tok']),
        }
        out_rows.append({**base, 'metric': 'MRC',          'value': row['mrc'], 'sig': sig})
        out_rows.append({**base, 'metric': 'Success Rate',  'value': row['sr'],  'sig': sig})
        out_rows.append({**base, 'metric': 'sig',           'value': sig,        'sig': sig})

    out_df = pd.DataFrame(
        out_rows,
        columns=['token', 'model', 'dataset', 'metric', 'value', 'position', 'n_tok', 'sig'],
    )
    out_df.to_csv(out_file, sep='\t', index=False)

    n_rows = len(out_df)
    n_expected = len(out_df.model.unique()) * len(out_df.token.unique()) * 2 * 3
    print(f'\nWritten {n_rows} rows to {out_file}  (expected {n_expected})')
    if n_rows != n_expected:
        print('WARNING: row count mismatch — some (token, model, dataset) combinations may be missing.')
        print('         Check that all 4 model jobs have finished before running this script.')


if __name__ == '__main__':
    Fire(main)
