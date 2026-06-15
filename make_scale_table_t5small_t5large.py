"""
make_scale_table_t5small_t5large.py

Generate one LaTeX table containing both monoT5-small and monoT5-large
results (DL19 + DL20) from aggregated rank-change TSV files.

USAGE:
    # If you already have one merged TSV containing both models:
    python make_scale_table_t5small_t5large.py \
        --run_file aggregated_t5small_t5large.tsv \
        --out_file table3_t5small_t5large.tex

    # If you have separate files, merge first:
    cat aggregated_t5small.tsv aggregated_t5large.tsv > aggregated_t5small_t5large.tsv
"""

import pandas as pd
from fire import Fire

MODEL_DICT = {
    "t5.small": r"monoT5$_\text{small}$",
    "t5.large": r"monoT5$_\text{large}$",
}

DATA_DICT = {
    "dl19": "DL19",
    "dl20": "DL20",
}

TOKEN_GROUPS = {
    "Prompt Tokens": ["true", "false", "relevant", "relevanttrue", "relevantfalse"],
    "Control Tokens": ["bar", "baz", "information", "informationbar", "informationbaz", "relevantbar", "informationtrue"],
    "Synonyms": ["pertinent", "significant", "related", "associated", "important"],
    "Sub-Words": ["relevancy", "relevance", "relevantly", "irrelevant"],
    "Misspellings": ["relevanty", "relevent", "trues", "falses"],
}

POSITIONS = {
    "start": "s",
    "end": "e",
    "random": "r",
}


def format_mrc(mrc, sr, p, r, colour_level, sig):
    sig_tok = r"\sig" if sig else r"\insig"
    colour_token = "pos" if mrc >= 0.0 else "neg"
    sign = "+" if mrc >= 0.0 else "-"
    if mrc == 0.0:
        sign = ""
    return (
        r"\cellcolor{" + colour_token + f"!{colour_level}" + "}"
        + f"${sign}{abs(mrc)}" + sig_tok
        + r"_{\color{gray}" + f"{abs(sr)}, {p}, {r}" + r"}$"
    )


def main(run_file: str, out_file: str):
    df = pd.read_csv(run_file, sep="\t", index_col=False)

    # Handle legacy aggregate files with dataset labels like dl19.gz
    if "dataset" in df.columns:
        df["dataset"] = df["dataset"].astype(str).str.replace(".gz", "", regex=False)

    df = df[df["model"].isin(MODEL_DICT.keys())].copy()
    if df.empty:
        raise ValueError(
            f"No rows found for models {list(MODEL_DICT.keys())} in {run_file}"
        )

    max_mrc = float(df[df.metric == "MRC"].value.max()) if (df.metric == "MRC").any() else 0.0
    print(f"Max MRC used for colour normalisation: {max_mrc}")

    def colour_combo(mrc, sr, p, r, sig=False):
        norm_val = min(round((abs(mrc) / max_mrc) * 50), 50) if max_mrc != 0 else 0
        return format_mrc(round(mrc, 1), round(sr * 100), p, r, norm_val, sig)

    # 1 token column + 4 metric columns = 5 columns total
    preamble = (
        r"\begin{table*}" + "\n"
        r"    \centering" + "\n"
        r"    \footnotesize" + "\n"
        r"    \setlength{\tabcolsep}{2pt}" + "\n"
        r"    \begin{tabular}{@{}lcccc@{}}"
    )

    header = r"    \toprule"
    columns = (
        r"    Token & "
        + " & ".join(
            r"\multicolumn{" + str(len(DATA_DICT)) + r"}{c}{" + model + r"}"
            for model in MODEL_DICT.values()
        )
        + r"\\"
    )
    datasets = (
        r"    & "
        + " & ".join(
            " & ".join(
                r"\multicolumn{1}{c}{" + data + r"}"
                for data in DATA_DICT.values()
            )
            for _ in MODEL_DICT
        )
        + r"\\"
    )

    total = [preamble, header, columns, r"    \midrule", datasets, r"    \midrule"]

    for group, tokens in TOKEN_GROUPS.items():
        total.append(r"    \midrule")
        total.append(r"    \multicolumn{5}{l}{" + group + r"}\\")
        total.append(r"    \midrule")

        for token in tokens:
            token_subset = df[df.token == token].copy()
            row = "    " + token + " & "

            for model_key in MODEL_DICT:
                model_subset = token_subset[token_subset.model == model_key].copy()
                if model_subset.empty:
                    row += "--- & --- & "
                    continue

                # Expect 2 datasets * 3 metrics (MRC, Success Rate, sig)
                expected_rows = len(DATA_DICT) * 3
                if len(model_subset) != expected_rows:
                    print(
                        f"WARNING: token={token}, model={model_key}: expected {expected_rows} rows, got {len(model_subset)}"
                    )

                for data_key in DATA_DICT:
                    data_subset = model_subset[model_subset.dataset == data_key].copy()
                    if data_subset.empty:
                        row += "--- & "
                        continue

                    mrc_rows = data_subset[data_subset.metric == "MRC"]
                    sr_rows = data_subset[data_subset.metric == "Success Rate"]
                    sig_rows = data_subset[data_subset.metric == "sig"]

                    if mrc_rows.empty or sr_rows.empty or sig_rows.empty:
                        row += "--- & "
                        continue

                    mrc = float(mrc_rows.value.values[0])
                    sr = float(sr_rows.value.values[0])
                    sig = bool(sig_rows.value.values[0])

                    pos_raw = str(model_subset.position.values[0])
                    p = POSITIONS.get(pos_raw, pos_raw)
                    r = str(model_subset.n_tok.values[0])

                    row += colour_combo(mrc, sr, p, r, sig) + " & "

            row = row.rstrip(" & ") + r"\\"
            total.append(row)

    total += [
        r"    \bottomrule",
        r"    \caption{monoT5-small and monoT5-large keyword-stuffing attack results (MRC / SR). "
        r"Positive MRC = attack boosted rank. \sig = significant (Bonferroni-corrected Wilcoxon).}",
        r"    \label{tab:t5small_t5large}",
        r"    \end{tabular}",
        r"\end{table*}",
    ]

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(total) + "\n")

    print(f"Written: {out_file}")


if __name__ == "__main__":
    Fire(main)
