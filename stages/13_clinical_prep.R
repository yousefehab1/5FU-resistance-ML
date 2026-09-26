# ==============================================================================
# 13_clinical_prep.R
#
# Download the five FOLFOX clinical cohorts from GEO, keep the FOLFOX-treated
# arm of each, ComBat-correct the four Affymetrix (GPL570) studies together,
# and derive the covariates: tumour purity, CMS subtype, MSI-like status.
# Scoring and statistics happen in 14_clinical_validation.py.
#
# Outputs (data/processed/):
#   clinical_expr_gpl570_combat.csv     genes x samples, 4 studies, ComBat-corrected
#   clinical_expr_gpl570_precombat.csv  same, before ComBat (for the batch PCA)
#   clinical_expr_gse104645.csv         the Agilent study, kept separate
#   clinical_covariates.csv             sample, study, platform, response, covariates
#
# Cohorts (each checked against its GEO title and summary):
#   GSE28702  (n=83,  GPL570)  mFOLFOX6 throughout. All included.
#   GSE19860  (n=40,  GPL570)  modified FOLFOX6; "FL_Responder" is the FOLFOX
#                              response call. Bevacizumab kept as a covariate.
#   GSE69657  (n=30,  GPL570)  neoadjuvant FOLFOX4. All included.
#   GSE72970  (n=124, GPL570)  mixed regimens; FOLFOX arms only -> 36.
#   GSE104645 (n=193, GPL6480) mixed regimens; FOLFOX arms only -> 104.
#                              Two-colour Agilent log-ratios, a different
#                              technology, so never pooled with the Affymetrix
#                              studies. Responder = CR + PR; "Not evaluated"
#                              excluded.
#
# Covariates:
#   Purity  ESTIMATE (Yoshihara 2013), per study on the pre-ComBat matrix.
#   CMS     CMScaller NTP, per study, in HGNC-symbol space.
#   MSI     no series reports a molecular MSI test, so CMScaller's MSI
#           expression template is used as an MSI-like proxy, reported as such.
#
# Run: Rscript stages/13_clinical_prep.R
# ==============================================================================

suppressMessages({
  library(GEOquery)
  library(Biobase)
  library(sva)
  library(CMScaller)
  library(estimate)
})

RAW  <- file.path("data", "raw", "geo_clinical")
PROC <- file.path("data", "processed")
dir.create(RAW, showWarnings = FALSE, recursive = TRUE)
dir.create(PROC, showWarnings = FALSE, recursive = TRUE)

banner <- function(x) cat("\n", strrep("=", 78), "\n", x, "\n", strrep("=", 78), "\n", sep = "")

# ------------------------------------------------------------------------
# Step 0: fetch series matrices + platform annotation (cached across runs)
# ------------------------------------------------------------------------
banner("FETCHING GEO SERIES (cached under data/raw/geo_clinical)")
fetch <- function(acc) getGEO(acc, GSEMatrix = TRUE, destdir = RAW, getGPL = FALSE)[[1]]

esets <- list(
  GSE28702  = fetch("GSE28702"),
  GSE19860  = fetch("GSE19860"),
  GSE72970  = fetch("GSE72970"),
  GSE69657  = fetch("GSE69657"),
  GSE104645 = fetch("GSE104645")
)
for (acc in names(esets)) cat(sprintf("  %-10s %s, %d samples, %d features\n",
    acc, annotation(esets[[acc]]), ncol(esets[[acc]]), nrow(esets[[acc]])))

get_probe_map <- function(gpl_id, symbol_col) {
  cache <- file.path(RAW, paste0(gpl_id, "_probe2symbol.rds"))
  if (file.exists(cache)) return(readRDS(cache))
  gpl <- getGEO(gpl_id, destdir = RAW)
  tbl <- Table(gpl)[, c("ID", symbol_col)]
  saveRDS(tbl, cache)
  tbl
}
map570  <- get_probe_map("GPL570",  "Gene Symbol")
map6480 <- get_probe_map("GPL6480", "GENE_SYMBOL")

first_symbol <- function(x) sub(" ///.*$", "", trimws(as.character(x)))

# ------------------------------------------------------------------------
# probe -> gene-symbol matrix, log2 as needed, dup-symbol collapse (mean)
# ------------------------------------------------------------------------
collapse_probes <- function(m, sym) {
  # rowsum() groups rows by `sym`, returning one row per unique symbol
  # (sorted, row-named by the symbol) -- divide by per-symbol probe counts
  # to turn the sums into means.
  counts <- table(sym)
  sums <- rowsum(m, group = sym)
  sums / as.numeric(counts[rownames(sums)])
}

to_symbol_matrix <- function(eset, probe_map, id_col = "ID", sym_col) {
  m <- exprs(eset)
  mx <- max(m, na.rm = TRUE)
  if (mx > 30) {                      # linear scale -> log2
    floor_val <- min(m[m > 0], na.rm = TRUE)
    m[m <= 0] <- floor_val
    m <- log2(m)
    cat("    log2-transformed (was linear scale, max =", round(mx, 1), ")\n")
  } else {
    cat("    already log-scale (max =", round(mx, 1), ")\n")
  }
  sym <- first_symbol(probe_map[[sym_col]][match(rownames(m), probe_map[[id_col]])])
  keep <- !is.na(sym) & sym != "" & sym != "---"
  m <- m[keep, , drop = FALSE]; sym <- sym[keep]
  collapse_probes(m, sym)
}

# ------------------------------------------------------------------------
# Step 1: per-series FOLFOX-arm filter + binary response label, checked
# against each series' own GEO summary text, not guessed from column names
# ------------------------------------------------------------------------
banner("PER-SERIES FOLFOX-ARM FILTER + RESPONSE LABEL")

extract_GSE28702 <- function(eset) {
  pd <- pData(eset)
  data.frame(sample_id = rownames(pd), study = "GSE28702",
             response = ifelse(pd[["mfolfox6:ch1"]] == "responder", 1L, 0L),
             regimen = "FOLFOX", bev_added = FALSE,
             row.names = rownames(pd))
}

extract_GSE19860 <- function(eset) {
  pd <- pData(eset)
  resp_raw <- as.character(pd[["treatment response:ch1"]])
  fl <- sub(",.*$", "", resp_raw)                       # FL_Responder / FL_Non_responder
  bev <- grepl("BV_Responder|BV_Non_reponder", resp_raw)
  data.frame(sample_id = rownames(pd), study = "GSE19860",
             response = ifelse(fl == "FL_Responder", 1L,
                         ifelse(fl == "FL_Non_responder", 0L, NA_integer_)),
             regimen = "FOLFOX", bev_added = bev,
             row.names = rownames(pd))
}

extract_GSE69657 <- function(eset) {
  pd <- pData(eset)
  data.frame(sample_id = rownames(pd), study = "GSE69657",
             response = ifelse(pd[["chemoresponse:ch1"]] == "responder", 1L, 0L),
             regimen = "FOLFOX", bev_added = FALSE,
             row.names = rownames(pd))
}

extract_GSE72970 <- function(eset) {
  pd <- pData(eset)
  reg <- as.character(pd[["regimen:ch1"]])
  keep <- reg %in% c("FOLFOX", "FOLFOX+BEVACIZUMAB")
  pd <- pd[keep, , drop = FALSE]
  status <- as.character(pd[["response status:ch1"]])
  data.frame(sample_id = rownames(pd), study = "GSE72970",
             response = ifelse(status == "R", 1L, ifelse(status == "NR", 0L, NA_integer_)),
             regimen = "FOLFOX", bev_added = grepl("BEVACIZUMAB", reg[keep]),
             row.names = rownames(pd))
}

extract_GSE104645 <- function(eset) {
  pd <- pData(eset)
  reg <- as.character(pd[["1st-line chemotherapy regimens:ch1"]])
  keep <- reg %in% c("FOLFOX", "FOLFOX+Bev")
  pd <- pd[keep, , drop = FALSE]
  best <- as.character(pd[["best response of 1st-line chemotherapy:ch1"]])
  resp <- ifelse(best %in% c("Complete response", "Partial response"), 1L,
           ifelse(best %in% c("Stable disease", "Progressive disease"), 0L, NA_integer_))
  data.frame(sample_id = rownames(pd), study = "GSE104645",
             response = resp, regimen = "FOLFOX", bev_added = grepl("Bev", reg[keep]),
             row.names = rownames(pd))
}

pheno570 <- rbind(
  extract_GSE28702(esets$GSE28702),
  extract_GSE19860(esets$GSE19860),
  extract_GSE69657(esets$GSE69657),
  extract_GSE72970(esets$GSE72970)
)
pheno570 <- pheno570[!is.na(pheno570$response), , drop = FALSE]
pheno6480 <- extract_GSE104645(esets$GSE104645)
pheno6480 <- pheno6480[!is.na(pheno6480$response), , drop = FALSE]

for (s in unique(pheno570$study))
  cat(sprintf("  %-10s n=%3d FOLFOX-arm, response 1/0 = %d/%d\n", s,
              sum(pheno570$study == s), sum(pheno570$study == s & pheno570$response == 1),
              sum(pheno570$study == s & pheno570$response == 0)))
cat(sprintf("  %-10s n=%3d FOLFOX-arm, response 1/0 = %d/%d  (GPL6480, kept separate)\n",
            "GSE104645", nrow(pheno6480), sum(pheno6480$response == 1), sum(pheno6480$response == 0)))
cat(sprintf("\n  Total GPL570 FOLFOX n = %d (plan estimated 'roughly 132' across all five "
            , nrow(pheno570)))
cat("series; actual is different once regimen and platform are checked against source -- see script docstring.)\n")

# ------------------------------------------------------------------------
# Step 2: build symbol matrices, restrict to the filtered FOLFOX samples
# ------------------------------------------------------------------------
banner("BUILDING PER-STUDY SYMBOL MATRICES")
cat("  GSE28702:\n");  m28702  <- to_symbol_matrix(esets$GSE28702,  map570, sym_col = "Gene Symbol")[, pheno570$sample_id[pheno570$study=="GSE28702"], drop=FALSE]
cat("  GSE19860:\n");  m19860  <- to_symbol_matrix(esets$GSE19860,  map570, sym_col = "Gene Symbol")[, pheno570$sample_id[pheno570$study=="GSE19860"], drop=FALSE]
cat("  GSE69657:\n");  m69657  <- to_symbol_matrix(esets$GSE69657,  map570, sym_col = "Gene Symbol")[, pheno570$sample_id[pheno570$study=="GSE69657"], drop=FALSE]
cat("  GSE72970:\n");  m72970  <- to_symbol_matrix(esets$GSE72970,  map570, sym_col = "Gene Symbol")[, pheno570$sample_id[pheno570$study=="GSE72970"], drop=FALSE]
cat("  GSE104645:\n"); m104645 <- to_symbol_matrix(esets$GSE104645, map6480, sym_col = "GENE_SYMBOL")[, pheno6480$sample_id, drop=FALSE]

# ------------------------------------------------------------------------
# Step 3: covariates, computed PER STUDY on its own pre-ComBat log2 matrix
# ------------------------------------------------------------------------
banner("COVARIATES: ESTIMATE PURITY (per study, pre-batch-correction)")

run_estimate <- function(m, platform, tag) {
  td <- tempfile(); dir.create(td)
  # filterCommonGenes() reads its input with read.table(header=TRUE,
  # row.names=1) starting at line 1 -- it does NOT expect the 3-line GCT
  # preamble that outputGCT() itself writes (verified directly: feeding it
  # an outputGCT() file silently corrupts sample names into read.table's
  # positional X/X.1/X.2 placeholders, an inconsistency inside the
  # `estimate` package between its own writer and its own reader). Feed it
  # a plain tab-delimited table instead, exactly as its own vignette does.
  plain   <- file.path(td, paste0(tag, ".txt"))
  gct_out <- file.path(td, paste0(tag, "_estimate.gct"))
  write.table(data.frame(GeneSymbol = rownames(m), as.data.frame(m), check.names = FALSE),
              plain, sep = "\t", row.names = FALSE, quote = FALSE)
  filtered <- file.path(td, paste0(tag, "_filtered.gct"))
  invisible(capture.output(estimate::filterCommonGenes(plain, filtered, id = "GeneSymbol")))
  invisible(capture.output(estimate::estimateScore(filtered, gct_out, platform = platform)))
  res <- read.delim(gct_out, skip = 2, check.names = FALSE)
  scores <- t(res[, -(1:2)]); colnames(scores) <- res$NAME
  scores <- as.data.frame(scores)
  scores$sample_id <- rownames(scores)
  colnames(scores)[colnames(scores) == "TumorPurity"] <- "purity"   # ESTIMATE's own purity column (Yoshihara 2013 eq. 4)
  if (!"purity" %in% colnames(scores)) {
    # estimateScore()'s own source: the cos() purity conversion is only
    # computed when platform == "affymetrix" -- for agilent/illumina it
    # returns Stromal/Immune/ESTIMATE scores with no calibrated purity at
    # all (the conversion formula was fit on Affymetrix TCGA data). Not a
    # bug to route around: report it as NA and say why, rather than
    # inventing a purity number ESTIMATE itself declines to produce.
    cat(sprintf("    %s (%s): ESTIMATE does not calibrate TumorPurity for this platform; purity = NA, ESTIMATEScore reported instead\n", tag, platform))
    scores$purity <- NA_real_
  }
  scores[, c("sample_id", "StromalScore", "ImmuneScore", "ESTIMATEScore", "purity")]
}

est28702  <- run_estimate(m28702,  "affymetrix", "GSE28702")
est19860  <- run_estimate(m19860,  "affymetrix", "GSE19860")
est69657  <- run_estimate(m69657,  "affymetrix", "GSE69657")
est72970  <- run_estimate(m72970,  "affymetrix", "GSE72970")
est104645 <- run_estimate(m104645, "agilent",    "GSE104645")
purity_all <- rbind(est28702, est19860, est69657, est72970, est104645)
cat(sprintf("  purity computed for %d samples (range %.2f - %.2f)\n",
            nrow(purity_all), min(purity_all$purity), max(purity_all$purity)))

banner("COVARIATES: CMS SUBTYPE (CMScaller, per study, symbol space)")
run_cms <- function(m, tag) {
  res <- CMScaller(m, rowNames = "symbol", doPlot = FALSE, verbose = FALSE)
  data.frame(sample_id = rownames(res), cms = as.character(res$prediction), study_tag = tag)
}
cms_all <- rbind(
  run_cms(m28702,  "GSE28702"),  run_cms(m19860,  "GSE19860"),
  run_cms(m69657,  "GSE69657"),  run_cms(m72970,  "GSE72970"),
  run_cms(m104645, "GSE104645")
)
cms_all$cms[is.na(cms_all$cms)] <- "NOLBL"   # CMScaller's own "not confidently classified" call
print(table(cms_all$cms))

banner("COVARIATES: MSI-LIKE NTP CLASSIFICATION (proxy -- no molecular MSI test in any of the 5 series)")
msi_template <- CMScaller::templates.MSI
msi_template$probe <- msi_template$symbol     # switch the template to symbol space (see docstring)
run_msi <- function(m, tag, fdr_max = 0.05) {
  common <- intersect(rownames(m), msi_template$probe)
  res <- CMScaller::ntp(m[common, , drop = FALSE], msi_template[msi_template$probe %in% common, ],
                         doPlot = FALSE, verbose = FALSE)
  # CMScaller() itself only accepts a template call below its FDR threshold
  # (default 0.05, "Indeterminate" otherwise); raw ntp() applies no such
  # threshold, so without one every sample gets a confident-looking MSI/MSS
  # label regardless of how weak the correlation to either template is.
  # Same confidence gate applied here, for the same reason.
  call <- as.character(res$prediction)
  call[res$FDR >= fdr_max] <- "Indeterminate"
  data.frame(sample_id = rownames(res), msi_ntp = call,
             msi_ntp_fdr = res$FDR, study_tag = tag)
}
msi_all <- rbind(
  run_msi(m28702,  "GSE28702"),  run_msi(m19860,  "GSE19860"),
  run_msi(m69657,  "GSE69657"),  run_msi(m72970,  "GSE72970"),
  run_msi(m104645, "GSE104645")
)
print(table(msi_all$msi_ntp))

# ------------------------------------------------------------------------
# Step 4: ComBat across the 4 GPL570 studies on their shared gene symbols
# ------------------------------------------------------------------------
banner("BATCH CORRECTION: ComBat across the 4 GPL570 studies (GSE104645 excluded, different platform)")
common_genes <- Reduce(intersect, list(rownames(m28702), rownames(m19860), rownames(m69657), rownames(m72970)))
cat(sprintf("  %d genes common to all 4 GPL570 studies\n", length(common_genes)))
combined570 <- cbind(m28702[common_genes, ], m19860[common_genes, ], m69657[common_genes, ], m72970[common_genes, ])
batch <- pheno570$study[match(colnames(combined570), pheno570$sample_id)]
combat570 <- sva::ComBat(dat = as.matrix(combined570), batch = batch, mod = NULL, par.prior = TRUE)
cat(sprintf("  ComBat done: %d genes x %d samples\n", nrow(combat570), ncol(combat570)))

# ------------------------------------------------------------------------
# Step 5: assemble + write outputs
# ------------------------------------------------------------------------
saveRDS(list(pheno570=pheno570, pheno6480=pheno6480, purity_all=purity_all,
             cms_all=cms_all, msi_all=msi_all, combat570=combat570, m104645=m104645),
        file.path(RAW, "checkpoint_covariates.rds"))

banner("WRITING OUTPUTS")

cov570 <- merge(pheno570, purity_all[, c("sample_id","purity","StromalScore","ImmuneScore")], by = "sample_id")
cov570 <- merge(cov570, cms_all[cms_all$study_tag != "GSE104645", c("sample_id","cms")], by = "sample_id")
cov570 <- merge(cov570, msi_all[msi_all$study_tag != "GSE104645", c("sample_id","msi_ntp","msi_ntp_fdr")], by = "sample_id")
cov570$platform <- "GPL570"
cov570 <- cov570[match(colnames(combat570), cov570$sample_id), ]

cov6480 <- merge(pheno6480, purity_all[purity_all$sample_id %in% pheno6480$sample_id, c("sample_id","purity","StromalScore","ImmuneScore")], by = "sample_id")
cov6480 <- merge(cov6480, cms_all[cms_all$study_tag == "GSE104645", c("sample_id","cms")], by = "sample_id")
cov6480 <- merge(cov6480, msi_all[msi_all$study_tag == "GSE104645", c("sample_id","msi_ntp","msi_ntp_fdr")], by = "sample_id")
cov6480$platform <- "GPL6480"
cov6480 <- cov6480[match(colnames(m104645), cov6480$sample_id), ]

write.csv(cbind(gene = rownames(combat570), as.data.frame(combat570)),
          file.path(PROC, "clinical_expr_gpl570_combat.csv"), row.names = FALSE)
write.csv(cbind(gene = rownames(combined570), as.data.frame(combined570)),
          file.path(PROC, "clinical_expr_gpl570_precombat.csv"), row.names = FALSE)
write.csv(cbind(gene = rownames(m104645), as.data.frame(m104645)),
          file.path(PROC, "clinical_expr_gse104645.csv"), row.names = FALSE)
write.csv(rbind(cov570, cov6480), file.path(PROC, "clinical_covariates.csv"), row.names = FALSE)

cat("  ->", file.path(PROC, "clinical_expr_gpl570_combat.csv"), "\n")
cat("  ->", file.path(PROC, "clinical_expr_gpl570_precombat.csv"), "(same genes/samples, before ComBat -- for the batch-structure PCA)\n")
cat("  ->", file.path(PROC, "clinical_expr_gse104645.csv"), "\n")
cat("  ->", file.path(PROC, "clinical_covariates.csv"), "\n")
banner("DONE")
