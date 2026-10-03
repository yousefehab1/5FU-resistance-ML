# =============================================================================
# export_gene_tss_hg38.R
#
# Export one TSS per gene on hg38 (TxDb.Hsapiens.UCSC.hg38.knownGene) for
# 11_regulatory_architecture.py. The TCGA peak-to-gene links cover only
# 11-23% of COAD peaks, so the remaining peaks are assigned to the nearest TSS.
#
# Per gene: symbol, chrom, tss, strand, gene_start, gene_end, gene_length.
# The TSS is the outermost 5' end; gene_length is used to match the
# permutation background.
#
# Run:    Rscript scripts/one_time/export_gene_tss_hg38.R
# Writes: data/raw/atac/gene_tss_hg38.csv
# =============================================================================

suppressMessages({
  library(TxDb.Hsapiens.UCSC.hg38.knownGene)
  library(org.Hs.eg.db)
  library(GenomicFeatures)
})

message("Building gene TSS/length table from TxDb.Hsapiens.UCSC.hg38.knownGene ...")

txdb <- TxDb.Hsapiens.UCSC.hg38.knownGene
g <- genes(txdb)  # one range per Entrez gene: outermost start/end across its transcripts

message(sprintf("  %d Entrez genes in the TxDb", length(g)))

entrez <- names(g)
sym <- suppressMessages(
  AnnotationDbi::mapIds(org.Hs.eg.db, keys = entrez, keytype = "ENTREZID",
                        column = "SYMBOL", multiVals = "first"))

df <- data.frame(
  entrez = entrez,
  symbol = unname(sym),
  chrom = as.character(GenomicRanges::seqnames(g)),
  gene_start = GenomicRanges::start(g),
  gene_end = GenomicRanges::end(g),
  strand = as.character(GenomicRanges::strand(g)),
  stringsAsFactors = FALSE
)
df <- df[!is.na(df$symbol), , drop = FALSE]

# TSS = 5' end: gene_start on +, gene_end on -
df$tss <- ifelse(df$strand == "+", df$gene_start, df$gene_end)
df$gene_length <- df$gene_end - df$gene_start + 1L

# Keep canonical chromosomes only (drop scaffolds/alt contigs/patches).
canonical <- paste0("chr", c(1:22, "X", "Y"))
df <- df[df$chrom %in% canonical, , drop = FALSE]

# A symbol can map to >1 Entrez gene (rare paralogs/readthroughs) -- keep the
# longest-length entry, not an arbitrary first row (never guess).
n_before <- nrow(df)
df <- df[order(df$symbol, -df$gene_length), ]
dup <- duplicated(df$symbol)
n_dropped <- sum(dup)
df <- df[!dup, , drop = FALSE]

message(sprintf("  %d gene rows resolved to a symbol on a canonical chromosome", n_before))
message(sprintf("  %d duplicate-symbol rows dropped (kept longest gene_length per symbol)", n_dropped))
message(sprintf("  %d unique gene symbols retained", nrow(df)))

out_dir <- file.path("data", "raw", "atac")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out_path <- file.path(out_dir, "gene_tss_hg38.csv")
write.csv(df[, c("symbol", "chrom", "tss", "strand", "gene_start", "gene_end", "gene_length")],
          out_path, row.names = FALSE)
message(sprintf("Wrote %s", out_path))
