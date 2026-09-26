# =============================================================================
# export_hgnc_aliases.R
#
# Export the HGNC alias -> official symbol table from org.Hs.eg.db.
#
# The HCT116 time course uses older gene symbols (SDPR, CYR61, CTGF ...) than
# the signatures and GDSC (CAVIN2, CCN1, CCN2 ...). Unresolved, DTP_UP would
# recover 91% of its genes in GDSC but only 67% in the time course, so the two
# arms would score different gene sets under the same name.
#
# An alias that points at more than one official symbol is dropped, never
# guessed; the number dropped is printed.
#
# Run:    Rscript scripts/export_hgnc_aliases.R
# Writes: data/raw/hgnc_alias_map.csv
# =============================================================================

if (!requireNamespace("org.Hs.eg.db", quietly = TRUE)) {
  stop("org.Hs.eg.db is required.\n",
       "Install with: BiocManager::install('org.Hs.eg.db')", call. = FALSE)
}

message("Building alias table from org.Hs.eg.db ...")

db  <- org.Hs.eg.db::org.Hs.eg.db
tbl <- suppressMessages(
  AnnotationDbi::select(db,
                        keys     = AnnotationDbi::keys(db, keytype = "ALIAS"),
                        keytype  = "ALIAS",
                        columns  = "SYMBOL"))

tbl <- tbl[!is.na(tbl$ALIAS) & !is.na(tbl$SYMBOL), , drop = FALSE]
tbl <- unique(tbl)
n_before <- nrow(tbl)

# Drop aliases that map to more than one official symbol (never guess).
n_targets <- table(tbl$ALIAS)
ambiguous <- names(n_targets)[n_targets > 1L]
tbl <- tbl[!tbl$ALIAS %in% ambiguous, , drop = FALSE]

message(sprintf("  %d alias->symbol pairs", n_before))
message(sprintf("  %d ambiguous aliases dropped (mapped to >1 official symbol)",
                length(ambiguous)))
message(sprintf("  %d unambiguous pairs retained", nrow(tbl)))

out_dir <- file.path("data", "raw")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out <- file.path(out_dir, "hgnc_alias_map.csv")

write.csv(data.frame(alias = tbl$ALIAS, symbol = tbl$SYMBOL),
          out, row.names = FALSE)

message(sprintf("Wrote %s", out))

# Spot-check the renames that motivated this, so a silent failure is visible.
check <- c(SDPR = "CAVIN2", CYR61 = "CCN1", CTGF = "CCN2",
           HIST1H4H = "H4C8", PIERCE1 = "C9orf116")
map <- stats::setNames(tbl$SYMBOL, tbl$ALIAS)
message("\nSpot-check:")
for (old in names(check)) {
  got <- unname(map[old])
  ok  <- !is.na(got) && got == check[[old]]
  message(sprintf("  %-10s -> %-10s expected %-10s %s",
                  old, ifelse(is.na(got), "NA", got), check[[old]],
                  ifelse(ok, "OK", "MISMATCH")))
}
