# module_12_stage1_step1_m4_per_channel_identity_test report

- Timestamp: 2026-05-22T13:30:45.215655
- Status: **PASS**
- Tolerance (bit-exact): 1e-07
- Windows tested: 24

## Per-channel max abs diff

- mae_MotSV: 5.96e-08
- mae_PmpSV: 1.49e-08
- mae_MotTV: 2.24e-08
- mae_PmpPV: 1.49e-08
- mae_TempSV: 5.96e-08
- mae_PresSV: 5.96e-08
- mae_PmpTV: 1.49e-08
- mae_MotPV: 2.98e-08

## Sample scale check (first window per label, patched values)


### Label 0

- mae_MotSV: 0.2653
- mae_PmpSV: 0.0970
- mae_MotTV: 0.0263
- mae_PmpPV: 0.0501
- mae_TempSV: 0.3456
- mae_PresSV: 0.2098
- mae_PmpTV: 0.1842
- mae_MotPV: 0.6093

### Label 21

- mae_MotSV: 0.0503
- mae_PmpSV: 0.0481
- mae_MotTV: 0.0704
- mae_PmpPV: 0.0277
- mae_TempSV: 0.2926
- mae_PresSV: 0.2464
- mae_PmpTV: 0.1588
- mae_MotPV: 0.0814

## Per-window detail

- seq_idx=4500 label=0 max_diff=2.98e-08 [PASS]
- seq_idx=4501 label=0 max_diff=5.96e-08 [PASS]
- seq_idx=4502 label=0 max_diff=5.96e-08 [PASS]
- seq_idx=0 label=1 max_diff=2.98e-08 [PASS]
- seq_idx=1 label=1 max_diff=1.49e-08 [PASS]
- seq_idx=2 label=1 max_diff=2.98e-08 [PASS]
- seq_idx=8000 label=3 max_diff=1.49e-08 [PASS]
- seq_idx=8001 label=3 max_diff=1.49e-08 [PASS]
- seq_idx=8002 label=3 max_diff=2.98e-08 [PASS]
- seq_idx=1500 label=4 max_diff=1.49e-08 [PASS]
- seq_idx=1501 label=4 max_diff=1.49e-08 [PASS]
- seq_idx=1502 label=4 max_diff=7.45e-09 [PASS]
- seq_idx=3000 label=5 max_diff=2.98e-08 [PASS]
- seq_idx=3001 label=5 max_diff=7.45e-09 [PASS]
- seq_idx=3002 label=5 max_diff=2.24e-08 [PASS]
- seq_idx=15200 label=10 max_diff=1.49e-08 [PASS]
- seq_idx=15201 label=10 max_diff=1.49e-08 [PASS]
- seq_idx=15202 label=10 max_diff=3.73e-09 [PASS]
- seq_idx=22100 label=15 max_diff=1.49e-08 [PASS]
- seq_idx=22101 label=15 max_diff=2.98e-08 [PASS]
- seq_idx=22102 label=15 max_diff=7.45e-09 [PASS]
- seq_idx=28900 label=21 max_diff=4.47e-08 [PASS]
- seq_idx=28901 label=21 max_diff=2.98e-08 [PASS]
- seq_idx=28902 label=21 max_diff=1.49e-08 [PASS]
