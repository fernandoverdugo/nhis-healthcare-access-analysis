# Raw NHIS files

The project uses two official 2025 NHIS public-use CSV files:

- `adult25.csv` — Sample Adult public-use file, 24,215 records
- `adultinc25.csv` — Sample Adult multiply imputed income file, 242,150 records (10 imputations per Sample Adult)

The raw CSV files are intentionally excluded from Git history.

## Automated download

From the repository root, run:

```bash
python scripts/download_data.py
```

The script downloads the official NCHS ZIP archives from:

- https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2025/adult25csv.zip
- https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2025/adultinc25csv.zip

It extracts the CSVs and verifies the NCHS-published MD5 checksums:

| File | MD5 |
|---|---|
| `adult25.csv` | `d8c359ca161e0bd3ef51345626308d72` |
| `adultinc25.csv` | `abb6a970181e852a4673ec591451f765` |

The official 2025 NHIS release and documentation are available at:
https://www.cdc.gov/nchs/nhis/documentation/2025-nhis.html
