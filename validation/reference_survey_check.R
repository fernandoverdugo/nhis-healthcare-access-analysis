suppressPackageStartupMessages(library(survey))

root <- normalizePath(getwd())
raw_file <- file.path(root, "data", "raw", "adult25.csv")
overall_file <- file.path(root, "data", "processed", "survey_overall.csv")
group_file <- file.path(root, "data", "processed", "survey_group_estimates.csv")

adult <- read.csv(raw_file, stringsAsFactors = FALSE)
adult$delayed_care_cost <- ifelse(
  adult$MEDDL12M_A == 1,
  1,
  ifelse(adult$MEDDL12M_A == 2, 0, NA)
)

options(survey.lonely.psu = "adjust")
design <- svydesign(
  id = ~PPSU,
  strata = ~PSTRAT,
  weights = ~WTFA_A,
  nest = TRUE,
  data = adult
)

python_overall <- read.csv(overall_file, stringsAsFactors = FALSE)
python_groups <- read.csv(group_file, stringsAsFactors = FALSE)

check_close <- function(label, actual, expected, tolerance = 1e-9) {
  difference <- abs(actual - expected)
  if (is.na(difference) || difference > tolerance) {
    stop(
      sprintf(
        "%s mismatch: R survey=%.12f, Python=%.12f, abs diff=%.3e",
        label,
        actual,
        expected,
        difference
      )
    )
  }
  cat(sprintf("Validated %-34s R=%.12f Python=%.12f\n", label, actual, expected))
}

extract_mean <- function(survey_design) {
  estimate <- svymean(~delayed_care_cost, design = survey_design, na.rm = TRUE)
  c(mean = as.numeric(coef(estimate)[1]), se = as.numeric(SE(estimate)[1]))
}

validate_group <- function(label, survey_design, dimension, level) {
  reference <- extract_mean(survey_design)
  python_row <- python_groups[
    python_groups$dimension == dimension & python_groups$level == level,
  ]
  if (nrow(python_row) != 1) {
    stop(sprintf("Expected one Python result for %s", label))
  }
  check_close(paste(label, "point estimate"), reference[["mean"]], python_row$weighted_prevalence)
  check_close(paste(label, "standard error"), reference[["se"]], python_row$standard_error)
}

overall_reference <- extract_mean(design)
check_close(
  "Overall point estimate",
  overall_reference[["mean"]],
  python_overall$weighted_prevalence[1]
)
check_close(
  "Overall standard error",
  overall_reference[["se"]],
  python_overall$standard_error[1]
)

if (degf(design) != python_overall$design_df[1]) {
  stop(
    sprintf(
      "Design degrees of freedom mismatch: R survey=%d, Python=%d",
      degf(design),
      python_overall$design_df[1]
    )
  )
}
cat(sprintf("Validated design degrees of freedom      R=%d Python=%d\n", degf(design), python_overall$design_df[1]))

validate_group(
  "Not covered",
  subset(design, NOTCOV_A == 1),
  "Insurance status",
  "Not covered"
)
validate_group(
  "No usual place of care",
  subset(design, USUALPL_A == 2),
  "Usual place of care",
  "No usual place"
)
validate_group(
  "Poor general health",
  subset(design, PHSTAT_A == 5),
  "General health",
  "Poor"
)

cat("Reference survey validation passed.\n")
