# Phase 5 - fusion verification

**ALL CHECKS PASSED**

| split | sequences | frames | in scope | positives | object rows | hard-braking clips | fusion hash |
|---|---|---|---|---|---|---|---|
| training | 800 | 42553 | 22923 | 5577 | 490988 | 564 | `830821bcfaec2be4` |
| validation | 95 | 5330 | 2803 | 710 | 61213 | 71 | `4cc6bea7dea90394` |
| test | 266 | 14125 | 7614 | 1862 | 164458 | 188 | `e2680f3a53cc6db9` |

## Checks

- PASS 7. no clip-time column among the model features
- PASS [training] 1. Phase 2 / Phase 4 inputs unchanged since fusion
- PASS [training] 2. frame_features.parquet intact
- PASS [training] 2. object_features.parquet intact
- PASS [training] 2. braking_events.csv intact
- PASS [training] 2. fusion hash matches the current settings (830821bcfaec2be4)
- PASS [training] 3. frame keys unique (42553 rows)
- PASS [training] 3. one frame row per Phase 2 row (42553 vs 42553)
- PASS [training] 3. split column
- PASS [training] 4. one object row per tracked box, same track ids (490988 vs 490988)
- PASS [training] 5. braking events recomputed independently (0 mismatches)
- PASS [training] 5. y consistent with onset and horizon
- PASS [training] 5. in_scope excludes warm-up and post-onset
- PASS [training] 5. lead_sec = onset - t
- PASS [training] 6. features unchanged when the future is removed (15 sequences)
- PASS [training] 8. attention in [0, 1]
- PASS [training] 8. contact time in (0, cap]
- PASS [training] 8. counts >= 0
- PASS [training] 8. no infinite feature value
- PASS [validation] 1. Phase 2 / Phase 4 inputs unchanged since fusion
- PASS [validation] 2. frame_features.parquet intact
- PASS [validation] 2. object_features.parquet intact
- PASS [validation] 2. braking_events.csv intact
- PASS [validation] 2. fusion hash matches the current settings (4cc6bea7dea90394)
- PASS [validation] 3. frame keys unique (5330 rows)
- PASS [validation] 3. one frame row per Phase 2 row (5330 vs 5330)
- PASS [validation] 3. split column
- PASS [validation] 4. one object row per tracked box, same track ids (61213 vs 61213)
- PASS [validation] 5. braking events recomputed independently (0 mismatches)
- PASS [validation] 5. y consistent with onset and horizon
- PASS [validation] 5. in_scope excludes warm-up and post-onset
- PASS [validation] 5. lead_sec = onset - t
- PASS [validation] 6. features unchanged when the future is removed (15 sequences)
- PASS [validation] 8. attention in [0, 1]
- PASS [validation] 8. contact time in (0, cap]
- PASS [validation] 8. counts >= 0
- PASS [validation] 8. no infinite feature value
- PASS [test] 1. Phase 2 / Phase 4 inputs unchanged since fusion
- PASS [test] 2. frame_features.parquet intact
- PASS [test] 2. object_features.parquet intact
- PASS [test] 2. braking_events.csv intact
- PASS [test] 2. fusion hash matches the current settings (e2680f3a53cc6db9)
- PASS [test] 3. frame keys unique (14125 rows)
- PASS [test] 3. one frame row per Phase 2 row (14125 vs 14125)
- PASS [test] 3. split column
- PASS [test] 4. one object row per tracked box, same track ids (164458 vs 164458)
- PASS [test] 5. braking events recomputed independently (0 mismatches)
- PASS [test] 5. y consistent with onset and horizon
- PASS [test] 5. in_scope excludes warm-up and post-onset
- PASS [test] 5. lead_sec = onset - t
- PASS [test] 6. features unchanged when the future is removed (15 sequences)
- PASS [test] 8. attention in [0, 1]
- PASS [test] 8. contact time in (0, cap]
- PASS [test] 8. counts >= 0
- PASS [test] 8. no infinite feature value
- PASS 9. splits are separate
