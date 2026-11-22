import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

NAN = float("nan")
# Same order, labels, colours and markers as plot_exp1.py, so a method has the same colour in every figure.
ORDER = ["gaussian", "spherical", "rademacher", "coordinate", "qr", "qr_reorth", "qr_f32_reorth", "householder", "permuted_householder", "butterfly"]

LABEL = {"coordinate": "coordinate", "qr": "QR", "qr_reorth": "QR [reorthogonalization]",
         "qr_f32_reorth": "QR [float 32 with reorthogonalization]", "householder": "householder",
         "permuted_householder": "permuted householder", "butterfly": "butterfly",
         "gaussian": "gaussian", "spherical": "spherical", "rademacher": "rademacher"}
LETTER = {"coordinate": "C", "qr": "Q", "qr_reorth": "O", "qr_f32_reorth": "F", "householder": "H",
          "permuted_householder": "P", "butterfly": "B", "gaussian": "G", "spherical": "S", "rademacher": "R"}

PALETTE = ["#0173b2", "#de8f05", "#029e73", "#d55e00", "#cc78bc", "#ca9161", "#fbafe4", "#949494", "#ece133", "#56b4e9"]
COLOR = {f: PALETTE[i] for i, f in enumerate(ORDER)}
MARKER = {f: m for f, m in zip(ORDER, "osD^vP*X<>")}
TEXT_COLS = ("function", "family", "dtype", "ls_params")

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.titlesize": 14, "axes.labelsize": 12,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 12, "grid.alpha": 0.5})


def read_rows(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            for k, v in r.items():
                if k in TEXT_COLS:
                    continue
                try:
                    r[k] = float(v)
                except TypeError:
                    r[k] = NAN
                except ValueError:                       # text column (e.g. status, part, method): keep it
                    r[k] = v if v else NAN
            r["d"], r["l"] = int(r["d"]), int(r["l"])
            rows.append(r)
    return rows


def l_label(l, d):
    if l == 2:
        return r"$\ell = 2$"
    for num, den, txt in ((1, 1, "d"), (4, 5, "4d/5"), (1, 2, "d/2"), (1, 3, "d/3")):
        if l == (num * d) // den:
            return rf"$\ell = {txt}$"
    return rf"$\ell = {l}$"


def family_legend(fig, fams, ncol=4):
    handles = [Line2D([0], [0], color=COLOR[f], marker=MARKER[f], lw=2, ms=6, label=LABEL[f]) for f in fams]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), ncol), frameon=False, handlelength=2.5)
    return 0.04 + 0.05 * ((len(handles) + ncol - 1) // ncol)        # bottom margin to leave for the legend


def common_selection(rows, functions, d, dtype, families, ls):

    present = list(dict.fromkeys(r["function"] for r in rows))
    functions = functions or present
    missing = [f for f in functions if f not in present]
    if missing:
        raise SystemExit(f"functions not in the CSV: {missing}  (available: {present})")
    d = d if d is not None else min(r["d"] for r in rows)
    dtype = dtype or rows[0]["dtype"]
    sub = [r for r in rows if r["d"] == d and r["dtype"] == dtype and r["function"] in functions]
    if not sub:
        raise SystemExit(f"no rows for d={d}, dtype={dtype}")
    fams_in = {r["family"] for r in sub}
    fams = [f for f in (families or ORDER) if f in fams_in]
    ls_in = {r["l"] for r in sub}
    ls = sorted(ls_in) if ls is None else [l for l in ls if l in ls_in]
    return functions, d, dtype, fams, ls, sub