# NHIS data use and attribution

This repository analyzes the **2025 National Health Interview Survey (NHIS)** public-use Sample Adult and Sample Adult Income files produced by the National Center for Health Statistics (NCHS), Centers for Disease Control and Prevention.

## Source

Official release page:

- https://www.cdc.gov/nchs/nhis/documentation/2025-nhis.html

Public-use data archives used by this project:

- `adult25csv.zip` — Sample Adult
- `adultinc25csv.zip` — Sample Adult multiply imputed income

The repository's `scripts/download_data.py` downloads those archives directly from the official NCHS FTP directory and verifies the extracted CSV files against the NCHS-published MD5 checksums.

## Requested attribution

NCHS requests that published material derived from the 2025 NHIS acknowledge the source. This repository uses:

> Data Source: National Center for Health Statistics, National Health Interview Survey, 2025.

NCHS also requests that published analyses make clear that analyses, interpretations, and conclusions are those of the author rather than NCHS.

## Data-use conditions

The NHIS public-use data are provided for statistical reporting and analysis. Users must not attempt to identify survey respondents or link the public-use data to individually identifiable records.

This repository does not contain direct identifiers and does not perform re-identification or record linkage. Raw source files and respondent-level derived intermediates are excluded from Git history.

## License scope

The repository's MIT License applies to original code and repository documentation only. It does not relicense or replace the terms governing the NHIS source data.
