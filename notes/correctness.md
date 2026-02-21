# Correctness Verification Report

Comparison of `origin/master` vs optimized branch across 62 diverse test cases.

## Summary

| Metric | Value |
|--------|-------|
| Total test cases | 62 |
| Passed | 62 |
| Failed | 0 |
| Count mismatches | 0 |
| Formula set mismatches | 0 |
| Error value mismatches (>1e-4 ppm) | 0 |
| RDBE value mismatches (>1e-4) | 0 |

## Test Coverage

| Category | Tests | Description |
|----------|-------|-------------|
| Basic masses | 7 | 100-612 Da, no filters |
| PPM tolerances | 5 | 1-20 ppm at 300 Da |
| Da tolerances | 4 | 0.001-0.05 Da at 300 Da |
| Charge states | 5 | -2 to +2 |
| Adducts | 4 | H, Na, -H, K with charge |
| RDBE filtering | 7 | Various ranges at 400 Da |
| Octet rule | 1 | Octet only at 400 Da |
| RDBE + octet | 3 | Combined at 400 Da |
| RDBE + octet + charge | 3 | z=-1, +1, +2 |
| RDBE + octet + adduct | 3 | H, Na, -H adducts |
| min/max counts (dict) | 3 | C>=5, C<=10/H<=20, both |
| min/max counts (string) | 2 | "C5", "C12H22O11" |
| Element sets | 3 | CHN, CHNO, CHNOPSClBrFI |
| Element sets + filters | 2 | CHN, CHNO with RDBE+octet |
| Edge: small masses | 2 | Water (18 Da), ethylene (28 Da) |
| Edge: zero results | 1 | 1 Da (no valid formulae) |
| Known molecules | 3 | Glucose, caffeine, ATP |
| max_results cap | 2 | 100 and 50 result limits |
| Large mass + tight RDBE | 1 | 750 Da, RDBE(5,15)+octet |
| Combined complex | 1 | Adduct + bounds + RDBE + octet |

## Detailed Results

| # | Test | n | Status | Max PPM diff | Max Da diff | Max RDBE diff |
|---|------|---|--------|-------------|------------|--------------|
| 1 | `CHNOPSClBrFI_m400_5ppm` | 6133 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 2 | `CHNOPS_m100.0_5ppm` | 0 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 3 | `CHNOPS_m150.05_5ppm` | 4 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 4 | `CHNOPS_m180.063_5ppm` | 8 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 5 | `CHNOPS_m18_5ppm` | 1 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 6 | `CHNOPS_m1_5ppm` | 0 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 7 | `CHNOPS_m250.1_5ppm` | 30 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 8 | `CHNOPS_m28_5ppm` | 1 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 9 | `CHNOPS_m300.0_5ppm` | 84 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 10 | `CHNOPS_m300_5ppm_adduct-H_z-1` | 73 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 11 | `CHNOPS_m300_5ppm_adductH_z1` | 74 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 12 | `CHNOPS_m300_5ppm_adductK_z1` | 36 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 13 | `CHNOPS_m300_5ppm_adductNa_z1` | 51 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 14 | `CHNOPS_m300_5ppm_maxC10H20` | 52 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 15 | `CHNOPS_m300_5ppm_maxC12H22O11_str` | 0 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 16 | `CHNOPS_m300_5ppm_minC3_maxC15H30` | 43 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 17 | `CHNOPS_m300_5ppm_minC5` | 38 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 18 | `CHNOPS_m300_5ppm_minC5_str` | 38 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 19 | `CHNOPS_m300_5ppm_z-1` | 73 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 20 | `CHNOPS_m300_5ppm_z-2` | 58 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 21 | `CHNOPS_m300_5ppm_z0` | 84 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 22 | `CHNOPS_m300_5ppm_z1` | 77 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 23 | `CHNOPS_m300_5ppm_z2` | 58 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 24 | `CHNOPS_m300_da0.001` | 56 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 25 | `CHNOPS_m300_da0.005` | 288 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 26 | `CHNOPS_m300_da0.01` | 570 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 27 | `CHNOPS_m300_da0.05` | 2775 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 28 | `CHNOPS_m300_ppm1.0` | 20 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 29 | `CHNOPS_m300_ppm10.0` | 169 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 30 | `CHNOPS_m300_ppm2.0` | 34 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 31 | `CHNOPS_m300_ppm20.0` | 344 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 32 | `CHNOPS_m300_ppm5.0` | 84 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 33 | `CHNOPS_m400_5ppm_octet` | 182 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 34 | `CHNOPS_m400_5ppm_rdbe(-0.5, 40)` | 203 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 35 | `CHNOPS_m400_5ppm_rdbe(0, 10)` | 115 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 36 | `CHNOPS_m400_5ppm_rdbe(0, 10)_octet` | 60 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 37 | `CHNOPS_m400_5ppm_rdbe(0, 15)` | 162 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 38 | `CHNOPS_m400_5ppm_rdbe(0, 20)` | 184 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 39 | `CHNOPS_m400_5ppm_rdbe(0, 20)_octet` | 94 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 40 | `CHNOPS_m400_5ppm_rdbe(0, 30)` | 200 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 41 | `CHNOPS_m400_5ppm_rdbe(0, 30)_octet` | 103 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 42 | `CHNOPS_m400_5ppm_rdbe(0, 5)` | 63 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 43 | `CHNOPS_m400_5ppm_rdbe(5, 15)` | 105 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 44 | `CHNOPS_m400_5ppm_rdbe0_20_octet_adduct-H_z-1` | 81 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 45 | `CHNOPS_m400_5ppm_rdbe0_20_octet_adductH_z1` | 81 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 46 | `CHNOPS_m400_5ppm_rdbe0_20_octet_adductNa_z1` | 58 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 47 | `CHNOPS_m400_5ppm_rdbe0_20_octet_z-1` | 81 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 48 | `CHNOPS_m400_5ppm_rdbe0_20_octet_z1` | 83 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 49 | `CHNOPS_m400_5ppm_rdbe0_20_octet_z2` | 80 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 50 | `CHNOPS_m500.0_5ppm` | 1063 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 51 | `CHNOPS_m500_5ppm_max100` | 100 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 52 | `CHNOPS_m500_5ppm_max50_rdbe0_20_octet` | 16 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 53 | `CHNOPS_m612.152_5ppm` | 2714 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 54 | `CHNOPS_m750_5ppm_rdbe5_15_octet` | 657 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 55 | `CHNO_m300_5ppm` | 6 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 56 | `CHNO_m300_5ppm_rdbe0_15_octet` | 1 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 57 | `CHN_m200_5ppm` | 0 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 58 | `CHN_m200_5ppm_rdbe0_10_octet` | 0 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 59 | `atp_exact` | 1142 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 60 | `caffeine_exact` | 10 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 61 | `combined_complex` | 43 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 62 | `glucose_exact` | 8 | PASS | 0.00e+00 | 0.00e+00 | 0.00e+00 |

## Conclusion

All 62 test cases produce **identical results** between `origin/master` and the optimized branch:

- Same candidate counts in every case
- Same formula sets (no missing or extra formulae)
- PPM errors match to <1e-4 ppm (rounding at serialization boundary)
- Da errors match to <1e-8 Da
- RDBE values match to <1e-4

The optimizations are purely performance improvements with no effect on correctness.
