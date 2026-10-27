# Dataset Notes

The project includes `data_penjualan.csv` with 80 rows. The original dataset source is not documented in the provided files; add it here if the source can be confirmed.

| Column | Description |
|---|---|
| `stok` | Available stock units |
| `terjual` | Units sold |
| `hari_ke` | Day index (1–30) |
| `cuaca` | Binary weather code; meaning undocumented |
| `hari_besar` | Special-day indicator (0/1) |
| `sisa_persen` | `(stok - terjual) / stok * 100`, rounded to 2 decimals |
| `label` | `0` = Safe, `1` = Potential Waste |

## Data Caveats

- Only 80 rows are available.
- `sisa_persen` is derived from other columns; training verifies that values match the formula.
- In the supplied data, labels are completely separated by `sisa_persen`: class 0 ranges from 0% to 32.23%, while class 1 ranges from 36.25% to 85.96%. The label may reflect a threshold rule, so high evaluation scores do not establish real-world generalization.
- No expiry dates, product identities, dated sales history, measured waste amounts, or dataset source are provided.

The original column names are retained to preserve compatibility with the trained legacy model. The app's CSV template and displayed table use English names where applicable.

Keep the existing column meanings and labels unchanged unless the dataset origin or label definition can be verified. For real use, obtain a larger documented dataset with a target grounded in observed waste.
