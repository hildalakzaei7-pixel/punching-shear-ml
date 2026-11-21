# Data dictionary — TU Dortmund punching shear database (Rev. 1, Nov 2022)

Source file: `20230131_PunchingShearDataRev1-Spyridis.xlsx` (sheet "Punching shear database Nov'22").
Dataset: Spyridis, P., Mellios, N., Uz, O., & Walter, L. (2023). *Database of experimental results from
punching shear tests with various types of reinforcement* [Dataset]. TU Dortmund.
https://doi.org/10.17877/DE290R-23085 — rights: "protected by original copyright"; **the raw file is not
redistributed in this repository**. Download it from the DOI and place it in `data/raw/`.

Layout: row 1 = descriptive header, row 2 = symbol/unit, row 3 blank, data from row 4.
The literature reference is written only on the first row of each test series; it is forward-filled
to give `source_id`, the grouping key for cross-validation.

| Column (sheet) | Code name | Unit | Notes |
|---|---|---|---|
| Literature reference | `reference` → `source_id` | – | 54 distinct sources in the full sheet |
| Designation | `specimen` | – | specimen label within the source |
| slab thickness, h | `h` | mm | |
| slab static height, h_ef | `d` | mm | effective depth; used as *d* in EC2 |
| column width, b | `b` | mm | side of square or diameter of circular column |
| (equivalent) column diameter, b' | `b_eq` | mm | equals 1.128·b for **every** row, circular columns included; not used. For circular columns `b` is taken as the diameter |
| column type | `column_type` | c / s | circle or square |
| mean cylinder compressive strength | `fc` | MPa | f_cm,cyl (already converted by the dataset authors) |
| reinforcement ratio, principal (x) | `rho_l` | % | flexural ratio; secondary direction (ρ2) is empty for all rows |
| longitudinal reinforcement strength | `fy` | MPa | missing for ~31% of usable plain-slab tests |
| s0, s1, ns, β, ρw, fyw | shear reinforcement | mm, –, °, %, MPa | used only to *exclude* shear-reinforced slabs |
| fibre % vol, fibre kg/m³ | fibre content | %, kg/m³ | used only to *exclude* fibre-reinforced slabs |
| radius of load application | `r_load` | mm | proxy for slab span / shear slenderness |
| Failure load, Vs | `V_test` | kN | target |
| Failure mode | `failure_mode` | S / F / blank | S = shear (punching), F = flexural |

## Profile of the file (computed, 21 Nov 2026)

* 543 tests from 54 sources (the dataset authors report 544 full-scale tests elsewhere; one row
  difference to be checked).
* 113 with shear reinforcement, 159 with fibres.
* **Plain slabs (no shear reinforcement, no fibres): 271 tests, 51 sources.**
  Failure mode: 141 punching (S), 9 flexural (F), 121 not stated.
* **Core modelling set** = plain slabs, excluding F and the manual exclusions below, with d, b, f_c, ρ_l,
  load radius and V_test present: **205 tests from 38 sources** (161 square, 44 circular columns),
  87 of them with unreported failure mode.
* No exact duplicate rows. No physically implausible f_c or ρ values.

## Cleaning decisions

1. Exclude shear-reinforced and fibre-reinforced slabs (different mechanism; later extension).
2. Exclude tests explicitly marked flexural (F).
3. Keep tests with blank failure mode (they include classic punching series such as Elstner & Hognestad
   1956 and Guandalini 2006), but flag them and report a sensitivity run without them.
4. `fy` is not a model input (21 % missing in the core set; EC2 v_Rd,c does not use it).
5. **Manual exclusions (44 rows)** after reading every reference title and specimen name. The sheet's
   columns describe these as plain slabs, but they are not:
   * Sissakis & Sheikh (2007): CFRP-strengthened specimens; only the three "Control" slabs are kept.
   * Binici & Bayrak (2003): CFRP strengthening study; the two specimens could not be confirmed as controls.
   * Siburg (2014), DF series: footing tests (r/d = 1.5–2.3), outside the flat-slab scope.
   * Theodorakopoulos & Swamy (1989): lightweight aggregate concrete.
   Meisami et al. (2013) was checked as well: its CFRP strengthening is coded as ρw, so the automatic
   filter already keeps only the two unstrengthened slabs.
6. Grouped cross-validation uses `source_id`: no test series appears in both training and test folds.
