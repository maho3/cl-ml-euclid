# Forward-calibration experiment ledger

| name | map | summary | b_fwd | quad_c2 | skew_med | predicted | d | PASS | note |
|---|---|---|---|---|---|---|---|---|---|
| v7_quadratic_mean | quadratic | mean | 0.31 | 0.30 | -0.30 | quad_c2=0.30 absorbed by quadratic map (modeled). cubic_c3=0... | 8.883 | FAIL | baseline v7 forward, quadratic map |
| cubic_mean | cubic | mean | 0.31 | 0.30 | -0.30 | quad_c2=0.30 absorbed by cubic map (modeled). skew_med=-0.30... | 1.748 | FAIL | cubic map, residual curvature |
| quad_median | quadratic | median | 0.33 | 0.32 | -0.30 | quad_c2=0.32 absorbed by quadratic map (modeled). cubic_c3=0... | 1.889 | FAIL | median summary for skew |
| cubic_median | cubic | median | 0.33 | 0.32 | -0.30 | quad_c2=0.32 absorbed by cubic map (modeled). z_slope=0.05 -... | 1.733 | FAIL | cubic+median |
| cubic_median_cal35 | cubic | median | 0.33 | 0.32 | -0.30 | quad_c2=0.32 absorbed by cubic map (modeled). z_slope=0.05 -... | 7.818 | FAIL | cubic+median, cal_frac=0.35 |
| cubic_median_cal50 | cubic | median | 0.33 | 0.32 | -0.30 | quad_c2=0.32 absorbed by cubic map (modeled). z_slope=0.05 -... | 8.062 | FAIL | cubic+median, cal_frac=0.50 |
| cubic_mean_cal50 | cubic | mean | 0.31 | 0.30 | -0.30 | quad_c2=0.30 absorbed by cubic map (modeled). skew_med=-0.30... | 8.147 | FAIL | cubic+mean, cal_frac=0.50 |
| hetero_quad_mean_cal50 | quadratic | mean | 0.31 | 0.30 | -0.30 | quad_c2=0.30 absorbed by quadratic map (modeled). cubic_c3=0... | 9.345 | FAIL | mass-dep scatter |
| hetero_quad_mean_cal20 | quadratic | mean | 0.31 | 0.30 | -0.30 | quad_c2=0.30 absorbed by quadratic map (modeled). cubic_c3=0... | 2.138 | FAIL | mass-dep scatter cal0.2 |
| mode_quad_cal50 | quadratic | mode | 0.35 | 0.31 | -0.30 | quad_c2=0.31 absorbed by quadratic map (modeled). cubic_c3=0... | 7.946 | FAIL | mode summary, last skew lever |
| bivariate_cal50 | quadratic | mean | 0.31 | 0.30 | -0.30 | quad_c2=0.30 absorbed by quadratic map (modeled). cubic_c3=0... | 1.364 | FAIL | bivariate (mu,ell), const cov, linear map |
| bivariate_quad_cal50 | quadratic | mean | 0.31 | 0.30 | -0.30 | quad_c2=0.30 absorbed by quadratic map (modeled). cubic_c3=0... | 1.158 | FAIL | bivariate, quadratic map, const cov |
| bivariate_lin_cal50_s2 | linear | mean | 0.31 | 0.30 | -0.30 | quad_c2=0.30 != 0 with a LINEAR map -> curved truth gives ma... | 0.535 | PASS | linear map seed2 (split scatter) |
| bivariate_quad_cal50_s2 | quadratic | mean | 0.31 | 0.30 | -0.30 | quad_c2=0.30 absorbed by quadratic map (modeled). cubic_c3=0... | 0.719 | PASS | quad map seed2 |
| bivariate_quad_cal50_s1 | quadratic | mean | 0.31 | 0.30 | -0.30 | quad_c2=0.30 absorbed by quadratic map (modeled). cubic_c3=0... | 1.181 | FAIL | quad seed1 |
| bivariate_quad_cal50_s3 | quadratic | mean | 0.31 | 0.30 | -0.30 | quad_c2=0.30 absorbed by quadratic map (modeled). cubic_c3=0... | 0.969 | PASS | quad seed3 |
| bivariate_quad_cal50_s4 | quadratic | mean | 0.31 | 0.30 | -0.30 | quad_c2=0.30 absorbed by quadratic map (modeled). cubic_c3=0... | 0.844 | PASS | quad seed4 |
| biv_quad_rholin_s0 | quadratic | mean | 0.31 | 0.30 | -0.30 | quad_c2=0.30 absorbed by quadratic map (modeled). cubic_c3=0... | 1.022 | FAIL | mass-dep rho, quad map, seed0 |
| biv_quad_rholin_s1 | quadratic | mean | 0.31 | 0.30 | -0.30 | quad_c2=0.30 absorbed by quadratic map (modeled). cubic_c3=0... | 1.158 | FAIL | mass-dep rho, quad map, seed1 |
| biv_quad_rholin_s2 | quadratic | mean | 0.31 | 0.30 | -0.30 | quad_c2=0.30 absorbed by quadratic map (modeled). cubic_c3=0... | 0.699 | PASS | mass-dep rho, quad map, seed2 |
| biv_quad_rholin_s3 | quadratic | mean | 0.31 | 0.30 | -0.30 | quad_c2=0.30 absorbed by quadratic map (modeled). cubic_c3=0... | 0.955 | PASS | mass-dep rho, quad map, seed3 |
| biv_quad_rholin_s4 | quadratic | mean | 0.31 | 0.30 | -0.30 | quad_c2=0.30 absorbed by quadratic map (modeled). cubic_c3=0... | 0.942 | PASS | mass-dep rho, quad map, seed4 |
| FINAL_bivariate_full | quadratic | mean | 0.31 | 0.30 | -0.30 | quad_c2=0.30 absorbed by quadratic map (modeled). cubic_c3=0... | 0.981 | PASS | WINNER: bivariate const-rho quad-map, full settings (500/2000/4), seed3 (median split) |
| FINAL_bivariate_linear_full | linear | mean | 0.31 | 0.30 | -0.30 | quad_c2=0.30 != 0 with a LINEAR map -> curved truth gives ma... | 0.879 | PASS | HEADLINE (linear map, simplicity): bivariate const-rho, full settings, seed3 |
| biv_lin_cal05_s0 | linear | mean | 0.31 | 0.30 | -0.30 | quad_c2=0.30 != 0 with a LINEAR map -> curved truth gives ma... | 1.682 | FAIL | cal_frac~0.05 (~100 clusters) |
| biv_lin_cal05_s1 | linear | mean | 0.31 | 0.30 | -0.30 | quad_c2=0.30 != 0 with a LINEAR map -> curved truth gives ma... | 0.748 | PASS | cal_frac~0.05 (~100 clusters) |
| biv_lin_cal05_s2 | linear | mean | 0.31 | 0.30 | -0.30 | quad_c2=0.30 != 0 with a LINEAR map -> curved truth gives ma... | 0.425 | PASS | cal_frac~0.05 (~100 clusters) |
| biv_lin_cal05_s3 | linear | mean | 0.31 | 0.30 | -0.30 | quad_c2=0.30 != 0 with a LINEAR map -> curved truth gives ma... | 1.421 | FAIL | cal_frac~0.05 (~100 clusters) |
| biv_lin_cal05_s4 | linear | mean | 0.31 | 0.30 | -0.30 | quad_c2=0.30 != 0 with a LINEAR map -> curved truth gives ma... | 1.523 | FAIL | cal_frac~0.05 (~100 clusters) |
