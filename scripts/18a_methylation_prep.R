# ==============================================================================
# 18a_methylation_prep.R
#
# Bioconductor half of the methylation ingest; 18_methylation_ingest.py
# finishes it.
#
# Source: GSE68379_Matrix.processed.txt.gz from GEO (processed 450K beta values
# at probe level). The GDSC1000 web resource that used to host this matrix
# returns HTTP 410; IDATs are not reprocessed.
#
# File quirk: the header has one field fewer than each data line (unlabelled
# row index), so columns are named manually from the header tokens.
#
# Each cell line has an AVG.Beta and a Detection.PVal column. Betas with
# detection p >= 0.01 are masked to NA before any aggregation. Cross-reactive,
# SNP and sex-chromosome probes are removed. Only cell lines that map to a
# SANGER_MODEL_ID are read, to keep memory under 16 GB.
#
# Run: Rscript scripts/18a_methylation_prep.R
# ==============================================================================

suppressMessages({
  library(data.table)
  library(IlluminaHumanMethylation450kanno.ilmn12.hg19)
  library(maxprobes)
})

RAW  <- file.path("data", "raw")
METH_RAW <- file.path(RAW, "methylation", "GSE68379_Matrix.processed.txt.gz")
PROC <- file.path("data", "processed")
dir.create(PROC, showWarnings = FALSE, recursive = TRUE)
banner <- function(x) cat("\n", strrep("=", 78), "\n", x, "\n", strrep("=", 78), "\n", sep = "")

DETECTION_P_MAX <- 0.01   # minfi's standard detection-call threshold

# ==============================================================================
# Step 0: map methylation sample names -> SANGER_MODEL_ID
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
sample_tokens <- header_line[-1]                     # 2056 tokens, header order
is_beta <- grepl("_AVG\\.Beta$", sample_tokens)
is_pval <- grepl("_Detection\\.PVal$", sample_tokens)
stopifnot(sum(is_beta) == sum(is_pval), sum(is_beta) + sum(is_pval) == length(sample_tokens))
cell_line_beta  <- sub("_AVG\\.Beta$", "", sample_tokens[is_beta])
cell_line_pval  <- sub("_Detection\\.PVal$", "", sample_tokens[is_pval])
stopifnot(identical(cell_line_beta, cell_line_pval))  # same order, paired 1:1

mapped_id <- lookup[norm_name(cell_line_beta)]
n_mapped <- sum(!is.na(mapped_id))
cat(sprintf("  %d / %d methylation cell lines mapped to a SANGER_MODEL_ID\n", n_mapped, length(cell_line_beta)))
cat(sprintf("  %d unmapped (kept out of the matrix; sample: %s)\n",
            sum(is.na(mapped_id)), paste(head(cell_line_beta[is.na(mapped_id)], 8), collapse = ", ")))

dup_ids <- unique(mapped_id[!is.na(mapped_id)][duplicated(mapped_id[!is.na(mapped_id)])])
cat(sprintf("  %d SANGER_MODEL_IDs hit by >1 methylation sample (averaged after masking): %s\n",
            length(dup_ids), paste(head(dup_ids, 5), collapse = ", ")))

# data-column index: header position k (1-indexed within sample_tokens, i.e.
# header_line position k+1) corresponds to data-file column k+2 (data has
# one extra leading unlabelled row-index column versus the header).
beta_pos_in_tokens <- which(is_beta)
pval_pos_in_tokens <- which(is_pval)
# align beta/pval token positions for each mapped cell line by name
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

# ==============================================================================
# Step 1: read only the needed columns (probe ID + mapped beta + mapped pval)
# ==============================================================================
banner("READING METHYLATION MATRIX (subset of columns, all 485,512 probes)")
t0 <- Sys.time()
dt <- fread(cmd = paste("gzcat", shQuote(METH_RAW)), header = FALSE, skip = 1,
            select = select_cols, col.names = select_names)
cat(sprintf("  read %d rows x %d cols in %.1f min\n", nrow(dt), ncol(dt),
            as.numeric(difftime(Sys.time(), t0, units = "mins"))))

# ==============================================================================
# Step 2: detection-p masking, then collapse to one column per model_id
# ==============================================================================
banner(sprintf("DETECTION-P MASKING (p >= %.2f -> NA) AND REPLICATE COLLAPSE", DETECTION_P_MAX))
beta_cols <- grep("^beta__", names(dt), value = TRUE)
pval_cols <- grep("^pval__", names(dt), value = TRUE)
n_masked <- 0L
n_total <- 0L
for (i in seq_along(beta_cols)) {
  bc <- beta_cols[i]; pc <- pval_cols[i]
  fail <- dt[[pc]] >= DETECTION_P_MAX
  fail[is.na(fail)] <- FALSE
  n_masked <- n_masked + sum(fail)
  n_total <- n_total + length(fail)
  if (any(fail)) set(dt, which(fail), bc, NA_real_)
}
cat(sprintf("  masked %d / %d beta calls (%.2f%%) at detection p >= %.2f\n",
            n_masked, n_total, 100 * n_masked / n_total, DETECTION_P_MAX))
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

# ==============================================================================
# Step 3: QC -- probe and sample missingness (post detection-p masking)
# ==============================================================================
banner("QC: PROBE AND SAMPLE MISSINGNESS")
probe_miss <- rowMeans(is.na(beta_mat))
sample_miss <- colMeans(is.na(beta_mat))
qc_probe <- data.table(probe_id = rownames(beta_mat), missing_frac = probe_miss)
qc_sample <- data.table(model_id = colnames(beta_mat), missing_frac = sample_miss)
cat(sprintf("  probe missingness: median %.4f, max %.4f\n", median(probe_miss), max(probe_miss)))
cat(sprintf("  sample missingness: median %.4f, max %.4f\n", median(sample_miss), max(sample_miss)))

# ==============================================================================
# Step 4: probe filters -- cross-reactive, SNP-overlapping, sex chromosomes
# ==============================================================================
banner("PROBE FILTERS: cross-reactive, SNP-overlapping, sex chromosomes")
ann <- as.data.frame(minfi::getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19))
ann <- ann[rownames(beta_mat), , drop = FALSE]
stopifnot(identical(rownames(ann), rownames(beta_mat)))

xr <- maxprobes::xreactive_probes(array_type = "450K")
is_xreactive <- rownames(beta_mat) %in% xr
is_snp <- !is.na(ann$Probe_rs) | !is.na(ann$CpG_rs) | !is.na(ann$SBE_rs)   # minfi::dropLociWithSnps default (maf=0: any annotated SNP)
is_sexchr <- ann$chr %in% c("chrX", "chrY")

cat(sprintf("  cross-reactive (maxprobes, Chen 2013 + Price 2013): %d\n", sum(is_xreactive)))
cat(sprintf("  SNP-overlapping (any Probe/CpG/SBE rs, minfi default maf=0): %d\n", sum(is_snp)))
cat(sprintf("  sex chromosome (chrX/chrY): %d\n", sum(is_sexchr)))
keep_probe <- !is_xreactive & !is_snp & !is_sexchr
cat(sprintf("  probes retained: %d / %d (%.1f%%)\n", sum(keep_probe), nrow(beta_mat), 100 * mean(keep_probe)))

beta_mat <- beta_mat[keep_probe, , drop = FALSE]
ann <- ann[keep_probe, , drop = FALSE]

# ==============================================================================
# Step 5: region aggregation -- promoter (TSS1500/TSS200) and gene body kept
# SEPARATE.
# ==============================================================================
banner("REGION AGGREGATION: promoter (TSS1500/TSS200) and gene body, kept separate")

split_probe_gene_region <- function(ann) {
  genes <- strsplit(ann$UCSC_RefGene_Name, ";")
  groups <- strsplit(ann$UCSC_RefGene_Group, ";")
  n <- lengths(genes)
  data.table(probe_id = rep(rownames(ann), n),
             gene = unlist(genes), region = unlist(groups))
}
long <- split_probe_gene_region(ann)
long <- unique(long[gene != "" & !is.na(gene)])

promoter_long <- unique(long[region %in% c("TSS1500", "TSS200"), .(probe_id, gene)])
body_long     <- unique(long[region == "Body", .(probe_id, gene)])
cat(sprintf("  promoter-associated probe-gene pairs: %d (%d unique genes)\n",
            nrow(promoter_long), length(unique(promoter_long$gene))))
cat(sprintf("  gene-body-associated probe-gene pairs: %d (%d unique genes)\n",
            nrow(body_long), length(unique(body_long$gene))))

aggregate_region <- function(region_long, beta_mat) {
  bm <- beta_mat[region_long$probe_id, , drop = FALSE]
  gene_beta <- rowsum(bm, group = region_long$gene, na.rm = TRUE)
  n_nonNA <- rowsum((!is.na(bm)) * 1L, group = region_long$gene)
  gene_beta / n_nonNA
}
promoter_beta <- aggregate_region(promoter_long, beta_mat)
body_beta     <- aggregate_region(body_long, beta_mat)
cat(sprintf("  promoter matrix: %d genes x %d samples\n", nrow(promoter_beta), ncol(promoter_beta)))
cat(sprintf("  body matrix:     %d genes x %d samples\n", nrow(body_beta), ncol(body_beta)))

# ==============================================================================
# Step 6: M-values for modelling, beta retained for interpretation
# ==============================================================================
banner("M-VALUE CONVERSION (M = log2(beta / (1 - beta)), beta clamped to [1e-4, 1-1e-4])")
to_M <- function(beta) {
  b <- pmin(pmax(beta, 1e-4), 1 - 1e-4)
  log2(b / (1 - b))
}
promoter_M <- to_M(promoter_beta)
body_M <- to_M(body_beta)

# ==============================================================================
# Step 7: write outputs (CSV here; scripts/18_methylation_ingest.py converts
# to parquet and does the expression-cohort overlap check and QC report).
# ==============================================================================
banner("WRITING INTERMEDIATE OUTPUTS")
write_gene_matrix <- function(m, path) {
  write.csv(cbind(gene = rownames(m), as.data.frame(m)), path, row.names = FALSE)
}
write_gene_matrix(promoter_beta, file.path(PROC, "methylation_promoter_beta_raw.csv"))
write_gene_matrix(promoter_M, file.path(PROC, "methylation_promoter_M_raw.csv"))
write_gene_matrix(body_beta, file.path(PROC, "methylation_body_beta_raw.csv"))
write_gene_matrix(body_M, file.path(PROC, "methylation_body_M_raw.csv"))
fwrite(qc_probe, file.path(PROC, "methylation_qc_probe_raw.csv"))
fwrite(qc_sample, file.path(PROC, "methylation_qc_sample_raw.csv"))

filter_counts <- data.table(
  filter = c("total_probes", "cross_reactive", "snp_overlap", "sex_chromosome", "retained"),
  n = c(length(keep_probe), sum(is_xreactive), sum(is_snp), sum(is_sexchr), sum(keep_probe))
)
fwrite(filter_counts, file.path(PROC, "methylation_probe_filter_counts.csv"))
fwrite(data.table(model_id = colnames(beta_mat)), file.path(PROC, "methylation_sample_ids.csv"))

cat(sprintf("  n_mapped_before_dup_collapse=%d, n_unique_model_id=%d, n_unmapped_cell_lines=%d\n",
            n_mapped, ncol(beta_mat), sum(is.na(mapped_id))))
cat("\n  -> ", file.path(PROC, "methylation_{promoter,body}_{beta,M}_raw.csv"), "\n")
cat("  -> ", file.path(PROC, "methylation_qc_{probe,sample}_raw.csv"), "\n")
cat("  -> ", file.path(PROC, "methylation_probe_filter_counts.csv"), "\n")

banner("DONE")
