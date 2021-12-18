import logging
import sys
import warnings

from anndata import AnnData
import pandas as pd
from scipy import sparse

from osn.preprocess.norm import norm_total, qc
from osn.preprocess.util import get_data_folders

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)
warnings.simplefilter(action="ignore", category=FutureWarning)


def load_expts(files, name=None, force=False, read_kwargs={}):
    data_fold = get_data_folders()
    if name is not None:
        files = {k: v for k, v in files.items() if k in name}
    for name, (count, meta) in files.items():
        adata_fn = data_fold.processed / f"{name}_norm.h5ad"
        if adata_fn.exists() and force is False:
            logging.info(
                f"Already found {adata_fn.name} file for {name}, skipping. To overwrite use `force=True`."
            )
        else:
            logging.info(f"Making {adata_fn} for {name}")
            make_new_obj(count, meta, adata_fn, **read_kwargs)


def make_new_obj(count, meta, adata_fn, **read_kwargs):
    logging.info("Loading count matrix and metadata. May take awhile.")
    df_meta = pd.read_csv(meta, index_col=0)
    df_count = pd.read_csv(count, index_col=0, **read_kwargs)
    logging.info("Loaded...now making adata object")
    adata = AnnData(df_count)
    del df_count
    assert df_meta.index.isin(adata.obs_names).all()
    # add metadata columns
    adata.obs = adata.obs.join(df_meta)
    if (adata.obs.orig_ident == "baseline-9").any():
        adata.obs["round2"] = adata.obs.orig_ident.isin(["baseline-9", "baseline-10"])
    umap_cols = adata.obs.columns.str.upper().str.contains("UMAP")
    if umap_cols.any():
        adata.obsm["X_umap"] = adata.obs.loc[:, umap_cols].values
    logging.info("Converting to sparse matrix")
    adata.X = sparse.csr_matrix(adata.X)
    qc(adata)
    adata.raw = adata.copy()
    logging.info("Normalizing total counts")
    adata = norm_total(adata)
    print(adata)
    logging.info(f"Saving to {adata_fn}")
    adata.write_h5ad(adata_fn)
    logging.info("Saved.")
