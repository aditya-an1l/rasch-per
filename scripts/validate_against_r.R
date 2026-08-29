#!/usr/bin/env Rscript
# Cross-validation of rasch-per item difficulties against R packages TAM and eRm.
#
# Requires R >= 4.0 with packages TAM and eRm installed:
#   install.packages(c("TAM", "eRm"))
#
# Usage:
#   Rscript scripts/validate_against_r.R responses.csv [--rasch-per-csv dif.csv]
#
# Fits a Rasch model with TAM::rasch.mml (and eRm::RM as a check), extracts
# item difficulties, and compares them to a CSV of rasch-per item difficulties
# (same item order) if provided.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript validate_against_r.R responses.csv [--rasch-per-csv dif.csv]")
}
csv <- args[1]
rp_csv <- NULL
if ("--rasch-per-csv" %in% args) {
  rp_csv <- args[which(args == "--rasch-per-csv") + 1]
}

if (!requireNamespace("TAM", quietly = TRUE)) stop("Please install the TAM package.")
if (!requireNamespace("eRm", quietly = TRUE)) stop("Please install the eRm package.")

dat <- as.matrix(read.csv(csv, row.names = 1))

tam <- TAM::rasch.mml(dat)
tam_beta <- tam$item$b

erm <- eRm::RM(dat)
erm_beta <- as.numeric(coef(erm))

cat("TAM item difficulties (logits):\n")
print(round(tam_beta, 4))
cat("\neRm item difficulties (logits):\n")
print(round(erm_beta, 4))

if (!is.null(rp_csv)) {
  rp <- read.csv(rp_csv)[, 1]
  comp <- data.frame(tam = tam_beta, eRm = erm_beta, rasch_per = rp)
  cat("\nCorrelation rasch-per vs TAM:", round(cor(comp$rasch_per, comp$tam), 4), "\n")
  cat("Correlation rasch-per vs eRm :", round(cor(comp$rasch_per, comp$eRm), 4), "\n")
  write.csv(comp, "r_cross_validation.csv", row.names = FALSE)
  cat("\nWrote r_cross_validation.csv\n")
}
