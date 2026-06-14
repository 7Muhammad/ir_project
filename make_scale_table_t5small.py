"""
make_scale_table_t5small.py

A t5.small-only variant of make_scale_table.py. Generates a LaTeX table
showing MRC and Success Rate for monoT5-small on DL19 and DL20.

Used to verify that the t5.small results match Table 3 of:
  "Analyzing Adversarial Attacks on Sequence-to-Sequence Relevance Models"
  ECIR 2024 — Parry, Fröbe, MacAvaney, Potthast, Hagen

USAGE:
    python make_scale_table_t5small.py \\
        --run_file aggregated_t5small.tsv \\
        --out_file table3_t5small.tex
"""

import pandas as pd
from fire import Fire

METRICS = ['MRC', 'Success Rate']

MODEL_DICT = {
    't5.small': r'monoT5$_\text{small}$',
}

DATA_DICT = {
    'dl19': 'DL19',
    'dl20': 'DL20',
}

TOKEN_GROUPS = {
    'Prompt Tokens':  ['true', 'false', 'relevant', 'relevanttrue', 'relevantfalse'],
    'Control Tokens': ['bar', 'baz', 'information', 'informationbar', 'informationbaz', 'relevantbar', 'informationtrue'],
    'Synonyms':       ['pertinent', 'significant', 'related', 'associated', 'important'],
    'Sub-Words':      ['relevancy', 'relevance', 'relevantly', 'irrelevant'],
    'Misspellings':   ['relevanty', 'relevent', 'trues', 'falses'],
}

POSITIONS = {
    'start':  's',
    'end':    'e',
    'random': 'r',
}


def format_mrc(mrc, sr, p, r, colour_level, sig):
    sig_tok = r'\sig' if sig else r'\insig'
    colour_token = 'pos' if mrc >= 0. else 'neg'
    sign = '+' if mrc >= 0. else '-'
    if mrc == 0.:
        sign = ''
    return (
        r'\cellcolor{' + colour_token + f'!{colour_level}' + '}'
        + f'${sign}{abs(mrc)}' + sig_tok
        + r'_{\color{gray}' + f'{abs(sr)}, {p}, {r}' + r'}$'
    )


def main(run_file: str, out_file: str):
    df = pd.read_csv(run_file, sep='\t', index_col=False)

    # Filter to t5.small only (aggregate_rank_changes may have other models if
    # this script is accidentally pointed at a full aggregated.tsv)
    df = df[df['model'] == 't5.small'].copy()
    if df.empty:
        raise ValueError(f"No t5.small rows found in {run_file}")

    max_vals = {
        'MRC':          df[df.metric == 'MRC'].value.max(),
        'Success Rate': df[df.metric == 'Success Rate'].value.max(),
    }
    print(f"Max values used for colour normalisation: {max_vals}")

    def colour_combo(mrc, sr, p, r, sig=False):
        max_val = float(max_vals['MRC'])
        norm_val = min(round((abs(mrc) / max_val) * 50), 50) if max_val != 0 else 0
        return format_mrc(round(mrc, 1), round(sr * 100), p, r, norm_val, sig)

    # LaTeX table: 3 columns — token label + DL19 + DL20
    preamble = (
        r'\begin{table}' + '\n'
        r'    \centering' + '\n'
        r'    \footnotesize' + '\n'
        r'    \setlength{\tabcolsep}{2pt}' + '\n'
        r'    \begin{tabular}{@{}lcc@{}}'
    )
    header   = r'    \toprule'
    columns  = (
        r'    Token & '
        + ' & '.join(
            r'\multicolumn{' + str(len(DATA_DICT)) + r'}{c}{' + model + r'}'
            for model in MODEL_DICT.values()
        )
        + r'\\'
    )
    datasets = (
        r'    & '
        + ' & '.join(
            ' & '.join(
                r'\multicolumn{1}{c}{' + data + r'}'
                for data in DATA_DICT.values()
            )
            for _ in MODEL_DICT
        )
        + r'\\'
    )
    total = [preamble, header, columns, r'    \midrule', datasets, r'    \midrule']

    for group, tokens in TOKEN_GROUPS.items():
        total.append(r'    \midrule')
        total.append(r'    \multicolumn{3}{l}{' + group + r'}\\')
        total.append(r'    \midrule')
        for token in tokens:
            token_subset = df[df.token == token].copy()
            row = '    ' + token + ' & '
            for model_key in MODEL_DICT:
                model_subset = token_subset[token_subset.model == model_key].copy()
                if model_subset.empty:
                    print(f'  WARNING: no data for token={token}, model={model_key}')
                    row += '--- & --- & '
                    continue
                assert len(model_subset) == len(DATA_DICT) * 3, (
                    f"Expected {len(DATA_DICT) * 3} rows for token={token}, model={model_key}, "
                    f"got {len(model_subset)}"
                )
                for data_key in DATA_DICT:
                    data_subset = model_subset[model_subset.dataset == data_key].copy()
                    mrc = float(data_subset[data_subset.metric == 'MRC'].value.values[0])
                    sr  = float(data_subset[data_subset.metric == 'Success Rate'].value.values[0])
                    sig = bool(data_subset[data_subset.metric == 'sig'].value.values[0])
                    pos = POSITIONS[model_subset.position.values[0]]
                    n   = str(model_subset.n_tok.values[0])
                    row += colour_combo(mrc, sr, pos, n, sig) + ' & '
            row = row.rstrip(' & ') + r'\\'
            total.append(row)

    total += [
        r'    \bottomrule',
        r'    \caption{monoT5-small keyword-stuffing attack results (MRC / SR). '
        r'Positive MRC = attack boosted rank. \sig = significant (Bonferroni-corrected Wilcoxon).}',
        r'    \label{tab:t5small}',
        r'    \end{tabular}',
        r'\end{table}',
    ]

    with open(out_file, 'w') as f:
        f.write('\n'.join(total) + '\n')

    print(f"\nWritten: {out_file}")


if __name__ == '__main__':
    Fire(main)
