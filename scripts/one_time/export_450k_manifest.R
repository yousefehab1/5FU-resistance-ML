# =============================================================================
# export_450k_manifest.R
#
# Export 450K probe annotation (probe ID, gene, UCSC_RefGene_Group,
# Relation_to_Island, enhancer flag) from
# IlluminaHumanMethylation450kanno.ilmn12.hg19, for stages/prep/methylation_context_prep.R.
#
# Run:    Rscript scripts/one_time/export_450k_manifest.R
# Writes: data/raw/methylation/humanmethylation450_manifest.csv
# =============================================================================

suppressMessages(library(IlluminaHumanMethylation450kanno.ilmn12.hg19))

message("Exporting HumanMethylation450 v1.2 (hg19) manifest ...")

ann <- as.data.frame(minfi::getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19))

out <- data.frame(
  probe_id = rownames(ann),
  chr = ann$chr,
  pos = ann$pos,
  gene = ann$UCSC_RefGene_Name,
  region = ann$UCSC_RefGene_Group,
  relation_to_island = ann$Relation_to_Island,
  enhancer = ann$Enhancer,
  regulatory_feature_group = ann$Regulatory_Feature_Group,
  dhs = ann$DHS,
  stringsAsFactors = FALSE
)

message(sprintf("  %d probes", nrow(out)))
message(sprintf("  %d flagged Enhancer=TRUE (%.2f%%)",
                sum(out$enhancer == "TRUE", na.rm = TRUE),
                100 * mean(out$enhancer == "TRUE", na.rm = TRUE)))
message("  Relation_to_Island distribution:")
print(table(out$relation_to_island))

out_dir <- file.path("data", "raw", "methylation")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out_path <- file.path(out_dir, "humanmethylation450_manifest.csv")
write.csv(out, out_path, row.names = FALSE)
message(sprintf("Wrote %s", out_path))
