# ==============================================================================
# methylation_context_prep.R
#
# R half of 09_methylation_context.py. Re-reads the probe-level 450K matrix
# with the same masking and probe filters as methylation_prep.R (which keeps only gene-level
# aggregates), then aggregates promoter probes separately per CpG-island
# relation (Island, N_Shore, S_Shore, N_Shelf, S_Shelf, OpenSea), plus an
# enhancer stratum (Enhancer = TRUE, any gene part).
#
# Needs data/raw/methylation/humanmethylation450_manifest.csv
# (export_450k_manifest.R).
#
# Run: Rscript stages/prep/methylation_context_prep.R
# ==============================================================================

suppressMessages({
  library(data.table)
  library(IlluminaHumanMethylation450kanno.ilmn12.hg19)
  library(maxprobes)
})

RAW  <- file.path("data", "raw")
METH_RAW <- file.path(RAW, "methylation", "GSE68379_Matrix.processed.txt.gz")
MANIFEST <- file.path(RAW, "methylation", "humanmethylation450_manifest.csv")
PROC <- file.path("data", "processed")
OUT_DIR <- file.path(PROC, "methylation_by_context")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
banner <- function(x) cat("\n", strrep("=", 78), "\n", x, "\n", strrep("=", 78), "\n", sep = "")

DETECTION_P_MAX <- 0.01

# ==============================================================================
# Steps 0-4: identical to methylation_prep.R (see its comments for the
# reasoning behind each choice) -- reproduced here because the filtered
# probe-level matrix isn't persisted by that script.
# ==============================================================================
banner("MAPPING METHYLATION CELL-LINE NAMES -> SANGER_MODEL_ID")
norm_name <- function(x) toupper(gsub("[^A-Za-z0-9]", "", x))
models <- fread(file.path(RAW, "model_list_20260724.csv"), select = c("model_id", "model_name", "synonyms"))
name_to_id <- setNames(models$model_id, norm_name(models$model_name))
syn <- models[!is.na(synonyms) & synonyms != "", .(model_id, synonyms)]
syn_long <- syn[, .(syn_name = trimws(unlist(strsplit(synonyms, ";")))), by = model_id]
syn_to_id <- setNames(syn_long$model_id, norm_name(syn_long$syn_name))
lookup <- c(name_to_id, syn_to_id[!norm_name(syn_long$syn_name) %in% names(name_to_id)])

header_line <- strsplit(readLines(gzfile(METH_RAW), n = 1), "\t")[[1]]
stopifnot(header_line[1] == "Row.names")
sample_tokens <- header_line[-1]
is_beta <- grepl("_AVG\\.Beta$", sample_tokens)
is_pval <- grepl("_Detection\\.PVal$", sample_tokens)
stopifnot(sum(is_beta) == sum(is_pval), sum(is_beta) + sum(is_pval) == length(sample_tokens))
cell_line_beta  <- sub("_AVG\\.Beta$", "", sample_tokens[is_beta])
cell_line_pval  <- sub("_Detection\\.PVal$", "", sample_tokens[is_pval])
stopifnot(identical(cell_line_beta, cell_line_pval))

mapped_id <- lookup[norm_name(cell_line_beta)]
n_mapped <- sum(!is.na(mapped_id))
cat(sprintf("  %d / %d methylation cell lines mapped to a SANGER_MODEL_ID\n", n_mapped, length(cell_line_beta)))
dup_ids <- unique(mapped_id[!is.na(mapped_id)][duplicated(mapped_id[!is.na(mapped_id)])])

beta_pos_in_tokens <- which(is_beta)
pval_pos_in_tokens <- which(is_pval)
keep_cl <- cell_line_beta[!is.na(mapped_id)]
keep_model_id <- mapped_id[!is.na(mapped_id)]
beta_tok_idx <- beta_pos_in_tokens[!is.na(mapped_id)]
pval_tok_idx <- pval_pos_in_tokens[match(keep_cl, cell_line_pval)]
beta_data_col <- beta_tok_idx + 2L
pval_data_col <- pval_tok_idx + 2L
probe_data_col <- 2L
uniq_tag <- paste0(keep_model_id, "__s", seq_along(keep_model_id))
select_cols <- c(probe_data_col, beta_data_col, pval_data_col)
select_names <- c("probe_id", paste0("beta__", uniq_tag), paste0("pval__", uniq_tag))

banner("READING METHYLATION MATRIX (subset of columns, all 485,512 probes)")
t0 <- Sys.time()
dt <- fread(cmd = paste("gzcat", shQuote(METH_RAW)), header = FALSE, skip = 1,
            select = select_cols, col.names = select_names)
cat(sprintf("  read %d rows x %d cols in %.1f min\n", nrow(dt), ncol(dt),
            as.numeric(difftime(Sys.time(), t0, units = "mins"))))

banner(sprintf("DETECTION-P MASKING (p >= %.2f -> NA) AND REPLICATE COLLAPSE", DETECTION_P_MAX))
beta_cols <- grep("^beta__", names(dt), value = TRUE)
pval_cols <- grep("^pval__", names(dt), value = TRUE)
n_masked <- 0L; n_total <- 0L
for (i in seq_along(beta_cols)) {
  bc <- beta_cols[i]; pc <- pval_cols[i]
  fail <- dt[[pc]] >= DETECTION_P_MAX
  fail[is.na(fail)] <- FALSE
  n_masked <- n_masked + sum(fail); n_total <- n_total + length(fail)
  if (any(fail)) set(dt, which(fail), bc, NA_real_)
}
cat(sprintf("  masked %d / %d beta calls (%.2f%%)\n", n_masked, n_total, 100 * n_masked / n_total))
dt[, (pval_cols) := NULL]

beta_mat <- as.matrix(dt[, ..beta_cols])
rownames(beta_mat) <- dt$probe_id
colnames(beta_mat) <- keep_model_id
rm(dt); gc()

if (length(dup_ids) > 0) {
  collapsed <- matrix(NA_real_, nrow = nrow(beta_mat), ncol = length(unique(colnames(beta_mat))),
                       dimnames = list(rownames(beta_mat), unique(colnames(beta_mat))))
  for (id in unique(colnames(beta_mat))) {
    cols <- which(colnames(beta_mat) == id)
    collapsed[, id] <- if (length(cols) == 1) beta_mat[, cols] else rowMeans(beta_mat[, cols, drop = FALSE], na.rm = TRUE)
  }
  beta_mat <- collapsed
  rm(collapsed); gc()
}
cat(sprintf("  final matrix: %d probes x %d unique SANGER_MODEL_IDs\n", nrow(beta_mat), ncol(beta_mat)))

banner("PROBE FILTERS: cross-reactive, SNP-overlapping, sex chromosomes")
ann <- as.data.frame(minfi::getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19))
ann <- ann[rownames(beta_mat), , drop = FALSE]
stopifnot(identical(rownames(ann), rownames(beta_mat)))
xr <- maxprobes::xreactive_probes(array_type = "450K")
is_xreactive <- rownames(beta_mat) %in% xr
is_snp <- !is.na(ann$Probe_rs) | !is.na(ann$CpG_rs) | !is.na(ann$SBE_rs)
is_sexchr <- ann$chr %in% c("chrX", "chrY")
keep_probe <- !is_xreactive & !is_snp & !is_sexchr
cat(sprintf("  probes retained: %d / %d (%.1f%%)\n", sum(keep_probe), nrow(beta_mat), 100 * mean(keep_probe)))
beta_mat <- beta_mat[keep_probe, , drop = FALSE]
ann <- ann[keep_probe, , drop = FALSE]

# ==============================================================================
# Step 5 (NEW): load the manifest for Relation_to_Island / Enhancer, join
# ==============================================================================
banner("LOADING MANIFEST (Relation_to_Island, Enhancer)")
manifest <- fread(MANIFEST)
manifest <- manifest[match(rownames(beta_mat), manifest$probe_id)]
stopifnot(identical(manifest$probe_id, rownames(beta_mat)))
cat(sprintf("  manifest matched for %d / %d retained probes\n",
            sum(!is.na(manifest$relation_to_island)), nrow(manifest)))

to_M <- function(beta) {
  b <- pmin(pmax(beta, 1e-4), 1 - 1e-4)
  log2(b / (1 - b))
}

aggregate_gene <- function(probe_ids, gene_for_probe, beta_mat) {
  bm <- beta_mat[probe_ids, , drop = FALSE]
  gene_beta <- rowsum(bm, group = gene_for_probe, na.rm = TRUE)
  n_nonNA <- rowsum((!is.na(bm)) * 1L, group = gene_for_probe)
  gene_beta / n_nonNA
}

write_gene_matrix <- function(m, path) {
  fwrite(cbind(gene = rownames(m), as.data.table(m)), path)
}

# region/gene long table, reused for both strata below
genes <- strsplit(ann$UCSC_RefGene_Name, ";")
groups <- strsplit(ann$UCSC_RefGene_Group, ";")
n <- lengths(genes)
long <- data.table(probe_id = rep(rownames(ann), n), gene = unlist(genes), region = unlist(groups))
long <- unique(long[gene != "" & !is.na(gene)])
long <- merge(long, manifest[, .(probe_id, relation_to_island, enhancer)], by = "probe_id", all.x = TRUE)

# ==============================================================================
# Step 6 (NEW, C2): promoter probes (TSS1500/TSS200), split by CpG-island
# relation -- Island / N_Shore / S_Shore / N_Shelf / S_Shelf / OpenSea
# ==============================================================================
banner("C2: PROMOTER PROBES, STRATIFIED BY CpG-ISLAND RELATION")
promoter_long <- unique(long[region %in% c("TSS1500", "TSS200")])
strata <- c("Island", "N_Shore", "S_Shore", "N_Shelf", "S_Shelf", "OpenSea")
coverage_rows <- list()
for (s in strata) {
  sub <- unique(promoter_long[relation_to_island == s, .(probe_id, gene)])
  cat(sprintf("  %-8s: %d probe-gene pairs, %d unique genes\n", s, nrow(sub), length(unique(sub$gene))))
  if (nrow(sub) == 0) next
  m_beta <- aggregate_gene(sub$probe_id, sub$gene, beta_mat)
  m_M <- to_M(m_beta)
  write_gene_matrix(m_beta, file.path(OUT_DIR, sprintf("promoter_%s_beta.csv", s)))
  write_gene_matrix(m_M, file.path(OUT_DIR, sprintf("promoter_%s_M.csv", s)))
  coverage_rows[[s]] <- data.table(stratum = s, n_probe_gene_pairs = nrow(sub), n_genes = length(unique(sub$gene)))
}

# ==============================================================================
# Step 7 (NEW, C1/C2): enhancer-flagged probes, ANY gene part
# ==============================================================================
banner("C2: ENHANCER-FLAGGED PROBES (any gene part)")
enh_long <- unique(long[enhancer == "TRUE", .(probe_id, gene)])
cat(sprintf("  enhancer-flagged: %d probe-gene pairs, %d unique genes\n",
            nrow(enh_long), length(unique(enh_long$gene))))
if (nrow(enh_long) > 0) {
  m_beta <- aggregate_gene(enh_long$probe_id, enh_long$gene, beta_mat)
  m_M <- to_M(m_beta)
  write_gene_matrix(m_beta, file.path(OUT_DIR, "enhancer_beta.csv"))
  write_gene_matrix(m_M, file.path(OUT_DIR, "enhancer_M.csv"))
  coverage_rows[["enhancer"]] <- data.table(stratum = "enhancer", n_probe_gene_pairs = nrow(enh_long),
                                             n_genes = length(unique(enh_long$gene)))
}

fwrite(rbindlist(coverage_rows), file.path(PROC, "methylation_context_coverage_strata.csv"))
fwrite(data.table(model_id = colnames(beta_mat)), file.path(PROC, "methylation_context_sample_ids.csv"))

banner("DONE")
cat("  -> ", OUT_DIR, "/promoter_<stratum>_{beta,M}.csv\n", sep = "")
cat("  -> ", OUT_DIR, "/enhancer_{beta,M}.csv\n", sep = "")
cat("  -> ", file.path(PROC, "methylation_context_coverage_strata.csv"), "\n")
