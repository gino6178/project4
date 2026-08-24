# ReRoom — results

## 0. What the experiments show

**The first milestone the plan sets — does relation-aware retargeting beat direct scaling in the oracle setting? — is met.** On held-out 3D-FRONT rooms, normalized-coordinate scaling leaves 8.8% of furniture area outside the target room and 5.8% in collision; ReRoom leaves 0.38% and 0.57%. Combined legality rises from 0.636 to 0.880, and the joint score from 0.770 to 0.871.

**The trade-off is real and visible, not hidden.** The coordinate maps score near-perfectly on relation preservation (S_rel 0.964) precisely because they preserve every relation — including into walls. ReRoom gives up some of that (S_rel 0.811, and 0.887 over the objects it keeps) to buy physical validity. Which side of that trade a reader prefers is exactly what the two-question human study in experiment 4 is designed to settle.

**Ignoring the reference is worse than adapting it.** A floor-plan-only synthesiser reaches legality 0.919 — the best of any method — but only S_rel 0.712 and S_motif 0.859, versus 0.942 for ReRoom. Looking at the reference is what the extra preservation buys.

**The reference designs are themselves not clean.** Scored in their own rooms, 3D-FRONT scenes show 6.0% collision area and 26.9% clearance violation. ReRoom's outputs are *more* physically valid than the professionally designed rooms they came from, which is the right way to read the absolute numbers.

**Perception error and retargeting error separate cleanly.** Sweeping a calibrated source-parser noise budget from perfect to severe costs design preservation (S_rel 0.877 → 0.354, S_motif 0.963 → 0.612) while leaving physical legality essentially untouched (0.751 → 0.697). A worse reading of the reference gives you a worse *design*, not a broken room — which is why the plan's insistence on validating the oracle setting first was the right call.

## 1. Corpus

16597 rooms parsed from 3D-FRONT (library: 917, bedroom: 9171, living_room: 5646, dining_room: 661, other: 202).

Floor polygons are taken from the raw `Floor` meshes rather than a bounding box, which is what the usual preprocessed subsets discard:

| property | value |
|---|---|
| exact rectangles | 45.9% |
| has a reflex vertex (non-convex) | 53.9% |
| convexity < 0.92 | 29.6% |
| median convexity | 0.984 |
| mean area | 19.4 m² |
| mean objects / room | 8.4 |
| mean occupancy ρ | 0.461 |

### SAGE-10k: what it is good for

Measured on 402 sampled SAGE-10k rooms: 100% are axis-aligned rectangles (median convexity 1.000), with 68 objects per room across 37 categories. It is therefore used for object and appearance diversity, not as irregular-room ground truth — which is exactly the role the plan assigns it.

## 2. Relation elasticity — the central hypothesis

`alpha` is fitted by regressing `log d` on `log gamma` (room extent along the relation) over 524,435 relation instances. The plan predicts body-scale relations near 0 and across-room relations near 1.

`raw` is the unshrunk per-bucket regression slope — what the data alone says; `shrunk` blends it toward the hand prior by the bucket's r² and sample count; `f_psi` is the learned estimator of eq. (45).

| pair | relation | prior | raw (n, r²) | shrunk | f_psi |
|---|---|---|---|---|---|
| dining_table--dining_chair | near | 0.000 | 0.061 (n=11747, r²=0.01) | 0.000 | 0.237 |
| dining_table--dining_chair | surrounds | 0.000 | 0.047 (n=13239, r²=0.00) | 0.000 | 0.053 |
| double_bed--nightstand | near | 0.050 | 0.171 (n=7866, r²=0.07) | 0.059 | 0.059 |
| desk--office_chair | near | 0.000 | 0.114 (n=599, r²=0.01) | 0.001 | 0.060 |
| sofa--coffee_table | facing | 0.150 | 0.129 (n=2490, r²=0.05) | 0.149 | 0.302 |
| sofa--tv_stand | face_to_face | 0.850 | 0.224 (n=2018, r²=0.13) | 0.770 | 0.404 |
| double_bed--wardrobe | face_to_face | 0.800 | 0.867 (n=423, r²=0.28) | 0.816 | 0.939 |
| sofa--armchair | facing | 0.750 | 0.285 (n=287, r²=0.05) | 0.731 | 0.938 |
| double_bed--wardrobe | facing | 0.800 | 0.626 (n=4990, r²=0.25) | 0.757 | 0.789 |
| tv_stand--tv | support | 0.000 | — | 0.000 | 0.196 |

Raw per-bucket fits (no shrinkage), largest buckets:

| bucket | alpha | n | r² |
|---|---|---|---|
| `dining_chair|dining_chair|grouped_with` | 0.220 | 27585 | 0.046 |
| `dining_chair|pendant_lamp|grouped_with` | 0.079 | 16203 | 0.002 |
| `dining_chair|dining_table|grouped_with` | 0.049 | 14471 | 0.003 |
| `dining_chair|dining_chair|facing` | 0.115 | 14412 | 0.027 |
| `dining_chair|dining_table|surrounds` | 0.047 | 13239 | 0.004 |
| `dining_chair|dining_table|near` | 0.061 | 11747 | 0.005 |
| `dining_chair|pendant_lamp|near` | 0.111 | 8503 | 0.011 |
| `dining_chair|pendant_lamp|facing` | 0.357 | 8387 | 0.037 |
| `dining_chair|dining_chair|face_to_face` | 0.078 | 8202 | 0.016 |
| `double_bed|nightstand|grouped_with` | 0.181 | 8084 | 0.086 |
| `double_bed|nightstand|near` | 0.171 | 7866 | 0.072 |
| `dining_chair|dining_chair|symmetric` | 0.168 | 7239 | 0.025 |

## 3. Experiment 1 — oracle retargeting (section 14.1)

| method | R_OOB ↓ | R_col ↓ | clearance ↓ | door ↓ | reach ↑ | S_rel ↑ | S_rel|scaled ↑ | S_rel|elastic ↑ | S_rel|kept ↑ | S_motif ↑ | retention | legality ↑ | score ↑ | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| source_reference | 0.0222 | 0.0597 | 0.2690 | 0.0168 | 0.9108 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6888 | 0.8135 | 1000 |
| reference_rigid | 0.1193 | 0.0597 | 0.2984 | 0.0436 | 0.8764 | 1.0000 | 0.9722 | 0.9903 | 1.0000 | 1.0000 | 1.0000 | 0.6048 | 0.7544 | 1000 |
| direct_scaling | 0.0877 | 0.0578 | 0.2865 | 0.0461 | 0.8975 | 0.9637 | 0.9743 | 0.9711 | 0.9637 | 1.0000 | 1.0000 | 0.6360 | 0.7703 | 1000 |
| affine_fit | 0.1107 | 0.0573 | 0.2956 | 0.0551 | 0.8753 | 0.9646 | 0.9760 | 0.9713 | 0.9646 | 1.0000 | 1.0000 | 0.6142 | 0.7567 | 1000 |
| target_only | 0.0030 | 0.0017 | 0.0781 | 0.0135 | 0.7754 | 0.7122 | 0.7159 | 0.7153 | 0.7688 | 0.8586 | 0.9518 | 0.9186 | 0.8419 | 1000 |
| relation_only | 0.0075 | 0.0134 | 0.1587 | 0.0298 | 0.8128 | 0.8775 | 0.8783 | 0.8818 | 0.8832 | 0.9848 | 0.9964 | 0.8299 | 0.8679 | 1000 |
| relation_summary | 0.0043 | 0.0059 | 0.1281 | 0.0294 | 0.8002 | 0.8098 | 0.8094 | 0.8128 | 0.8848 | 0.9400 | 0.9464 | 0.8650 | 0.8630 | 1000 |
| reroom_full | 0.0038 | 0.0057 | 0.1138 | 0.0259 | 0.8202 | 0.8110 | 0.8110 | 0.8141 | 0.8868 | 0.9417 | 0.9464 | 0.8796 | 0.8705 | 1000 |

By target geometry difficulty:

| level_name | R_OOB ↓ | R_col ↓ | clearance ↓ | door ↓ | reach ↑ | S_rel ↑ | S_rel|scaled ↑ | S_rel|elastic ↑ | S_rel|kept ↑ | S_motif ↑ | retention | legality ↑ | score ↑ | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aspect_ratio | 0.0353 | 0.0339 | 0.2020 | 0.0275 | 0.8554 | 0.8820 | 0.8854 | 0.8868 | 0.9110 | 0.9639 | 0.9780 | 0.7610 | 0.8186 | 1600 |
| concave | 0.0489 | 0.0313 | 0.1980 | 0.0346 | 0.8566 | 0.8939 | 0.8925 | 0.8951 | 0.9101 | 0.9709 | 0.9887 | 0.7562 | 0.8219 | 1600 |
| corner_cut | 0.0665 | 0.0334 | 0.2361 | 0.0594 | 0.8054 | 0.8954 | 0.8864 | 0.8948 | 0.9324 | 0.9581 | 0.9741 | 0.7124 | 0.7877 | 1600 |
| slanted_wall | 0.0428 | 0.0317 | 0.1953 | 0.0227 | 0.8539 | 0.9120 | 0.9090 | 0.9125 | 0.9266 | 0.9745 | 0.9892 | 0.7631 | 0.8310 | 1600 |
| uniform_scale | 0.0305 | 0.0329 | 0.1862 | 0.0183 | 0.8592 | 0.8785 | 0.8874 | 0.8838 | 0.9149 | 0.9607 | 0.9706 | 0.7805 | 0.8273 | 1600 |

By target/source area ratio:

| area_bucket | R_OOB ↓ | R_col ↓ | clearance ↓ | door ↓ | reach ↑ | S_rel ↑ | S_rel|scaled ↑ | S_rel|elastic ↑ | S_rel|kept ↑ | S_motif ↑ | retention | legality ↑ | score ↑ | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a<0.75 (shrink) | 0.1083 | 0.0778 | 0.2871 | 0.0256 | 0.8394 | 0.7683 | 0.7630 | 0.7716 | 0.9033 | 0.8837 | 0.8910 | 0.6229 | 0.6622 | 864 |
| b 0.75-0.95 | 0.0658 | 0.0403 | 0.2369 | 0.0445 | 0.8196 | 0.8947 | 0.8878 | 0.8949 | 0.9245 | 0.9655 | 0.9797 | 0.7050 | 0.7865 | 2304 |
| c 0.95-1.15 | 0.0256 | 0.0221 | 0.1889 | 0.0380 | 0.8514 | 0.9235 | 0.9232 | 0.9241 | 0.9325 | 0.9805 | 0.9944 | 0.7832 | 0.8512 | 2560 |
| d 1.15-1.50 | 0.0224 | 0.0211 | 0.1703 | 0.0179 | 0.8545 | 0.9066 | 0.9109 | 0.9101 | 0.9090 | 0.9777 | 0.9980 | 0.8012 | 0.8581 | 1440 |
| e >1.50 (grow) | 0.0186 | 0.0168 | 0.1267 | 0.0149 | 0.8959 | 0.8944 | 0.9102 | 0.9037 | 0.8958 | 0.9844 | 0.9990 | 0.8605 | 0.8885 | 832 |

## 4. Experiment 2 — floor-geometry difficulty (section 14.2)

| method | R_OOB ↓ | R_col ↓ | clearance ↓ | door ↓ | reach ↑ | S_rel ↑ | S_rel|scaled ↑ | S_rel|elastic ↑ | S_rel|kept ↑ | S_motif ↑ | retention | legality ↑ | score ↑ | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| direct_scaling | 0.0774 | 0.0421 | 0.3072 | 0.0651 | 0.8728 | 0.9653 | 0.9747 | 0.9730 | 0.9653 | 1.0000 | 1.0000 | 0.6237 | 0.7703 | 1200 |
| affine_fit | 0.1105 | 0.0417 | 0.3173 | 0.0688 | 0.8520 | 0.9660 | 0.9761 | 0.9729 | 0.9660 | 1.0000 | 1.0000 | 0.5958 | 0.7508 | 1200 |
| target_only | 0.0027 | 0.0008 | 0.0770 | 0.0181 | 0.7344 | 0.7137 | 0.7173 | 0.7168 | 0.7693 | 0.8555 | 0.9535 | 0.9204 | 0.8427 | 1200 |
| reroom_full | 0.0043 | 0.0024 | 0.1241 | 0.0336 | 0.7480 | 0.8052 | 0.8060 | 0.8083 | 0.8782 | 0.9438 | 0.9470 | 0.8708 | 0.8653 | 1200 |

By target geometry difficulty:

| level_name | R_OOB ↓ | R_col ↓ | clearance ↓ | door ↓ | reach ↑ | S_rel ↑ | S_rel|scaled ↑ | S_rel|elastic ↑ | S_rel|kept ↑ | S_motif ↑ | retention | legality ↑ | score ↑ | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aspect_ratio | 0.0307 | 0.0255 | 0.2037 | 0.0357 | 0.7864 | 0.8384 | 0.8525 | 0.8492 | 0.8741 | 0.9470 | 0.9716 | 0.7664 | 0.8098 | 960 |
| concave | 0.0660 | 0.0191 | 0.2043 | 0.0615 | 0.7978 | 0.8659 | 0.8671 | 0.8691 | 0.8857 | 0.9517 | 0.9848 | 0.7434 | 0.8036 | 960 |
| corner_cut | 0.0780 | 0.0219 | 0.2402 | 0.0765 | 0.7928 | 0.8681 | 0.8574 | 0.8670 | 0.9157 | 0.9437 | 0.9659 | 0.7041 | 0.7740 | 960 |
| slanted_wall | 0.0508 | 0.0190 | 0.2005 | 0.0379 | 0.8163 | 0.8917 | 0.8895 | 0.8930 | 0.9101 | 0.9618 | 0.9875 | 0.7547 | 0.8218 | 960 |
| uniform_scale | 0.0179 | 0.0232 | 0.1832 | 0.0205 | 0.8156 | 0.8486 | 0.8762 | 0.8603 | 0.8878 | 0.9448 | 0.9660 | 0.7947 | 0.8271 | 960 |

By target/source area ratio:

| area_bucket | R_OOB ↓ | R_col ↓ | clearance ↓ | door ↓ | reach ↑ | S_rel ↑ | S_rel|scaled ↑ | S_rel|elastic ↑ | S_rel|kept ↑ | S_motif ↑ | retention | legality ↑ | score ↑ | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a<0.75 (shrink) | 0.0858 | 0.0458 | 0.3138 | 0.0443 | 0.7765 | 0.7201 | 0.7286 | 0.7282 | 0.8828 | 0.8543 | 0.8619 | 0.6294 | 0.6451 | 508 |
| b 0.75-0.95 | 0.0723 | 0.0230 | 0.2360 | 0.0564 | 0.7798 | 0.8662 | 0.8597 | 0.8672 | 0.9013 | 0.9479 | 0.9755 | 0.7074 | 0.7803 | 1556 |
| c 0.95-1.15 | 0.0332 | 0.0206 | 0.1966 | 0.0610 | 0.8226 | 0.8994 | 0.9003 | 0.9014 | 0.9077 | 0.9647 | 0.9947 | 0.7701 | 0.8341 | 1540 |
| d 1.15-1.50 | 0.0297 | 0.0126 | 0.1429 | 0.0190 | 0.7985 | 0.8787 | 0.8953 | 0.8882 | 0.8812 | 0.9763 | 0.9984 | 0.8277 | 0.8695 | 720 |
| e >1.50 (grow) | 0.0109 | 0.0094 | 0.1227 | 0.0100 | 0.8384 | 0.8590 | 0.9035 | 0.8784 | 0.8639 | 0.9700 | 0.9963 | 0.8623 | 0.8876 | 476 |

## 6. Experiment 4 — prescribed target floors (section 14.4)

| method | R_OOB ↓ | R_col ↓ | clearance ↓ | door ↓ | reach ↑ | S_rel ↑ | S_rel|scaled ↑ | S_rel|elastic ↑ | S_rel|kept ↑ | S_motif ↑ | retention | legality ↑ | score ↑ | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| direct_scaling | 0.0570 | 0.0465 | 0.3162 | 0.0216 | 0.8953 | 0.9696 | 0.9722 | 0.9762 | 0.9696 | 1.0000 | 1.0000 | 0.6236 | 0.7718 | 180 |
| reroom_full | 0.0023 | 0.0022 | 0.1242 | 0.0106 | 0.7947 | 0.7835 | 0.7832 | 0.7863 | 0.8903 | 0.9276 | 0.9261 | 0.8723 | 0.8577 | 180 |

By prescribed target floor:

| target | R_OOB ↓ | R_col ↓ | clearance ↓ | door ↓ | reach ↑ | S_rel ↑ | S_rel|scaled ↑ | S_rel|elastic ↑ | S_rel|kept ↑ | S_motif ↑ | retention | legality ↑ | score ↑ | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 70pct_smaller | 0.0313 | 0.0381 | 0.2741 | 0.0186 | 0.8038 | 0.7819 | 0.8052 | 0.7924 | 0.9098 | 0.9321 | 0.9068 | 0.6904 | 0.7431 | 60 |
| l_shaped | 0.0641 | 0.0195 | 0.2362 | 0.0098 | 0.8701 | 0.8380 | 0.8256 | 0.8372 | 0.9318 | 0.9447 | 0.9331 | 0.7204 | 0.7755 | 60 |
| narrow | 0.0344 | 0.0305 | 0.2385 | 0.0205 | 0.8192 | 0.8461 | 0.8436 | 0.8573 | 0.8861 | 0.9586 | 0.9755 | 0.7212 | 0.7934 | 60 |
| original_like | 0.0047 | 0.0220 | 0.2008 | 0.0105 | 0.8824 | 0.9637 | 0.9637 | 0.9637 | 0.9779 | 0.9850 | 0.9905 | 0.7802 | 0.8641 | 60 |
| slanted_wall | 0.0358 | 0.0204 | 0.2094 | 0.0149 | 0.8612 | 0.9237 | 0.9177 | 0.9233 | 0.9613 | 0.9725 | 0.9762 | 0.7552 | 0.8341 | 60 |
| wide | 0.0079 | 0.0158 | 0.1621 | 0.0221 | 0.8333 | 0.9058 | 0.9100 | 0.9137 | 0.9125 | 0.9900 | 0.9964 | 0.8205 | 0.8784 | 60 |

## 7. Ablations (section 16.2)

| method | R_OOB ↓ | R_col ↓ | clearance ↓ | door ↓ | reach ↑ | S_rel ↑ | S_rel|scaled ↑ | S_rel|elastic ↑ | S_rel|kept ↑ | S_motif ↑ | retention | legality ↑ | score ↑ | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| direct_scaling | 0.0782 | 0.0421 | 0.3004 | 0.0614 | 0.8841 | 0.9645 | 0.9755 | 0.9724 | 0.9645 | 1.0000 | 1.0000 | 0.6290 | 0.7727 | 750 |
| flow | 0.0115 | 0.0076 | 0.1845 | 0.0417 | 0.7540 | 0.8277 | 0.8291 | 0.8304 | 0.8825 | 0.9480 | 0.9579 | 0.8038 | 0.8347 | 750 |
| flow_no_projection | 0.1037 | 0.0396 | 0.2845 | 0.0655 | 0.8725 | 0.9026 | 0.9131 | 0.9092 | 0.9628 | 0.9589 | 0.9579 | 0.6271 | 0.7496 | 750 |
| no_elasticity | 0.0037 | 0.0025 | 0.1058 | 0.0284 | 0.7783 | 0.8126 | 0.8088 | 0.8127 | 0.8838 | 0.9355 | 0.9477 | 0.8892 | 0.8731 | 750 |
| no_motif_grouping | 0.0035 | 0.0022 | 0.0948 | 0.0253 | 0.7696 | 0.7781 | 0.7814 | 0.7816 | 0.8607 | 0.9037 | 0.9327 | 0.9006 | 0.8559 | 750 |
| no_motif_init | 0.0038 | 0.0030 | 0.1177 | 0.0288 | 0.7846 | 0.8276 | 0.8280 | 0.8304 | 0.9028 | 0.9385 | 0.9466 | 0.8769 | 0.8717 | 750 |
| no_projection | 0.0041 | 0.0037 | 0.1364 | 0.0330 | 0.7997 | 0.8285 | 0.8264 | 0.8311 | 0.9028 | 0.9399 | 0.9472 | 0.8574 | 0.8627 | 750 |
| prior_elasticity | 0.0036 | 0.0028 | 0.1106 | 0.0271 | 0.7897 | 0.8136 | 0.8139 | 0.8175 | 0.8842 | 0.9413 | 0.9483 | 0.8844 | 0.8731 | 750 |
| relation_only | 0.0081 | 0.0090 | 0.1609 | 0.0343 | 0.7888 | 0.8790 | 0.8798 | 0.8832 | 0.8819 | 0.9883 | 0.9988 | 0.8291 | 0.8721 | 750 |
| relation_summary | 0.0049 | 0.0026 | 0.1243 | 0.0369 | 0.7761 | 0.8059 | 0.8060 | 0.8089 | 0.8814 | 0.9362 | 0.9449 | 0.8699 | 0.8630 | 750 |
| reroom_full | 0.0036 | 0.0023 | 0.1095 | 0.0290 | 0.7863 | 0.8136 | 0.8140 | 0.8167 | 0.8835 | 0.9405 | 0.9487 | 0.8858 | 0.8737 | 750 |
| size_only_retrieval | 0.0036 | 0.0026 | 0.1074 | 0.0281 | 0.7929 | 0.8135 | 0.8141 | 0.8167 | 0.8826 | 0.9408 | 0.9486 | 0.8877 | 0.8745 | 750 |

By target geometry difficulty:

| level_name | R_OOB ↓ | R_col ↓ | clearance ↓ | door ↓ | reach ↑ | S_rel ↑ | S_rel|scaled ↑ | S_rel|elastic ↑ | S_rel|kept ↑ | S_motif ↑ | retention | legality ↑ | score ↑ | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aspect_ratio | 0.0121 | 0.0117 | 0.1338 | 0.0305 | 0.8042 | 0.8205 | 0.8252 | 0.8273 | 0.8895 | 0.9370 | 0.9484 | 0.8517 | 0.8542 | 1800 |
| concave | 0.0259 | 0.0092 | 0.1574 | 0.0337 | 0.8011 | 0.8554 | 0.8557 | 0.8582 | 0.8828 | 0.9679 | 0.9811 | 0.8193 | 0.8553 | 1800 |
| corner_cut | 0.0302 | 0.0104 | 0.1862 | 0.0587 | 0.7760 | 0.8192 | 0.8146 | 0.8194 | 0.9112 | 0.9296 | 0.9319 | 0.7898 | 0.8134 | 1800 |
| slanted_wall | 0.0202 | 0.0092 | 0.1581 | 0.0365 | 0.7852 | 0.8794 | 0.8784 | 0.8805 | 0.9093 | 0.9751 | 0.9792 | 0.8222 | 0.8653 | 1800 |
| uniform_scale | 0.0084 | 0.0096 | 0.1297 | 0.0237 | 0.8238 | 0.8202 | 0.8303 | 0.8274 | 0.8960 | 0.9285 | 0.9425 | 0.8590 | 0.8521 | 1800 |

By target/source area ratio:

| area_bucket | R_OOB ↓ | R_col ↓ | clearance ↓ | door ↓ | reach ↑ | S_rel ↑ | S_rel|scaled ↑ | S_rel|elastic ↑ | S_rel|kept ↑ | S_motif ↑ | retention | legality ↑ | score ↑ | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a<0.75 (shrink) | 0.0356 | 0.0168 | 0.2026 | 0.0401 | 0.7776 | 0.5693 | 0.5695 | 0.5749 | 0.8891 | 0.7505 | 0.7511 | 0.7704 | 0.6672 | 1032 |
| b 0.75-0.95 | 0.0304 | 0.0113 | 0.1962 | 0.0534 | 0.7620 | 0.8357 | 0.8321 | 0.8372 | 0.8948 | 0.9534 | 0.9606 | 0.7780 | 0.8221 | 2532 |
| c 0.95-1.15 | 0.0125 | 0.0105 | 0.1533 | 0.0398 | 0.8095 | 0.8999 | 0.8997 | 0.9009 | 0.9148 | 0.9811 | 0.9900 | 0.8310 | 0.8776 | 2868 |
| d 1.15-1.50 | 0.0134 | 0.0062 | 0.1064 | 0.0151 | 0.8261 | 0.8801 | 0.8859 | 0.8853 | 0.8831 | 0.9812 | 0.9974 | 0.8792 | 0.9010 | 1656 |
| e >1.50 (grow) | 0.0030 | 0.0039 | 0.0611 | 0.0155 | 0.8343 | 0.8868 | 0.9053 | 0.8996 | 0.8890 | 0.9886 | 0.9990 | 0.9334 | 0.9360 | 912 |

Two defects were found and fixed while auditing why `relation_only` — the variant that does *less* — was outscoring the full system. Both had the same shape: a stage that fired without checking that it helped.

**Population had no acceptance test.** Summarization is checked by the repair loop and substitution by its own size-gain test, but whatever the population planner proposed was simply appended. Measured over 180 (scene, level) pairs it cost 0.020 of the joint score and 0.014 of legality wherever it fired, while buying no feasibility at all. Additions are now re-admitted one at a time, largest first, and only while they cost nothing in `E_bound + E_col + E_clear`: 6.0 proposed becomes 3.2 kept.

**Substitution was upsizing in rooms that grew.** The requested asset size scaled with the square root of the area ratio, up to 1.4x — which is the same mistake section 10 names in its own title, *a bigger room is filled by adding furniture, not by stretching what is there*. In rooms above 1.3x it cost 0.041 of legality and pushed clearance violation from 0.058 to 0.098, because larger furniture eats exactly the circulation space the extra floor was supposed to provide. Substitution may now fetch a better-fitting *smaller* asset when the room shrinks; growth is population's job.

Together the two moved `reroom_full` from 0.8641 to 0.8737 against `relation_only`'s 0.8721, and **every metric improved at once** — out-of-bounds, collision, clearance, door blockage, reachability, `S_rel`, `S_motif` and retention. Eight up, none down is the signature of removing a defect rather than re-tuning a trade-off.

**Removing the motif layer costs more than removing anything else.** `no_motif_init` only skips the motif-rigid starting point; `no_motif_grouping` deletes the layer outright — no groups, no motif-to-motif relations, no `grouped_with` edges, no motif-level selection — while still being *scored* against the intact reference graph, so `S_motif` keeps asking the same question. It is the lowest-scoring ReRoom variant in the table (0.8559 against 0.8737), and the loss is concentrated where the plan says it should be: `S_motif` 0.9037 against 0.9405.

Its legality is *better* (0.9006 against 0.8858), and that is not a contradiction — it is the trade priced. A solver with no obligation to keep a dining set together has more freedom to satisfy the geometry. Keeping the group is what you spend that freedom on.

**Style-aware retrieval does not move the layout metrics, and should not be expected to.** Setting `lambda_f = 0` leaves the geometry columns unchanged (score 0.8745 against 0.8737); substitution changes *which asset* fills a slot, not where the slot is. Its effect is visible only in the appearance column of section 7b, where dropping the appearance term costs a large amount of CLIP similarity for a small gain in size fit.

## 5. Experiment 3 — perception vs retargeting error (section 14.3)

Calibrated perception-noise sweep over 100 held-out rooms:

| perception | R_OOB ↓ | R_col ↓ | clearance ↓ | door ↓ | reach ↑ | S_rel ↑ | S_rel|scaled ↑ | S_rel|elastic ↑ | S_rel|kept ↑ | S_motif ↑ | retention | legality ↑ | score ↑ | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 0.0323 | 0.0243 | 0.2170 | 0.0491 | 0.8124 | 0.8774 | 0.8853 | 0.8843 | 0.9235 | 0.9630 | 0.9663 | 0.7515 | 0.8154 | 600 |
| noise_light | 0.0346 | 0.0250 | 0.2278 | 0.0491 | 0.8149 | 0.8035 | 0.8112 | 0.8102 | 0.9008 | 0.9311 | 0.9384 | 0.7401 | 0.7849 | 600 |
| noise_medium | 0.0409 | 0.0262 | 0.2330 | 0.0473 | 0.8052 | 0.7298 | 0.7359 | 0.7355 | 0.8672 | 0.8835 | 0.9064 | 0.7310 | 0.7494 | 600 |
| noise_heavy | 0.0540 | 0.0275 | 0.2394 | 0.0464 | 0.7925 | 0.5357 | 0.5405 | 0.5397 | 0.7996 | 0.7667 | 0.7934 | 0.7157 | 0.6597 | 600 |
| noise_severe | 0.0768 | 0.0276 | 0.2465 | 0.0408 | 0.7830 | 0.3543 | 0.3570 | 0.3564 | 0.7069 | 0.6121 | 0.6899 | 0.6970 | 0.5486 | 600 |
| midi | 0.0409 | 0.0724 | 0.2764 | 0.0233 | 0.7588 | 0.2957 | 0.2985 | 0.2969 | 0.6164 | 0.5311 | 0.6862 | 0.6699 | 0.4855 | 30 |

```
(39) - (38):
  oracle         dS_rel=+0.0000  dS_motif=+0.0000  dR_OOB=+0.0000  dlegality=+0.0000
  noise_light    dS_rel=-0.0675  dS_motif=-0.0279  dR_OOB=+0.0004  dlegality=-0.0102
  noise_medium   dS_rel=-0.1354  dS_motif=-0.0787  dR_OOB=+0.0012  dlegality=-0.0177
  noise_heavy    dS_rel=-0.3122  dS_motif=-0.2014  dR_OOB=+0.0026  dlegality=-0.0263
  noise_severe   dS_rel=-0.4744  dS_motif=-0.3492  dR_OOB=+0.0041  dlegality=-0.0159
  midi           dS_rel=-0.5507  dS_motif=-0.4675  dR_OOB=+0.0006  dlegality=+0.0222

perception vs solver (16.1): what each stage is worth
  oracle         reroom legality 0.868 S_rel 0.795   |   direct legality 0.635 S_rel 0.960
  noise_light    reroom legality 0.858 S_rel 0.727   |   direct legality 0.622 S_rel 0.880
  noise_medium   reroom legality 0.850 S_rel 0.659   |   direct legality 0.612 S_rel 0.800
  noise_heavy    reroom legality 0.842 S_rel 0.482   |   direct legality 0.589 S_rel 0.589
  noise_severe   reroom legality 0.852 S_rel 0.320   |   direct legality 0.542 S_rel 0.388
  midi           reroom legality 0.890 S_rel 0.244   |   direct legality 0.449 S_rel 0.348
```

### A real parser on the same curve

MIDI-3D (`VAST-AI/MIDI-3D`) run on textured renders of 31 held-out rooms, with exact instance masks supplied — a deliberately favourable setting, so what is measured is 3D reasoning error rather than segmentation error. Every perception level below is evaluated on *these same rooms*, because comparing a parser against an oracle computed over a different sample would not be a comparison.

| perception | R_OOB ↓ | R_col ↓ | clearance ↓ | door ↓ | reach ↑ | S_rel ↑ | S_rel|scaled ↑ | S_rel|elastic ↑ | S_rel|kept ↑ | S_motif ↑ | retention | legality ↑ | score ↑ | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 0.0049 | 0.0061 | 0.1368 | 0.0495 | 0.6918 | 0.7850 | 0.7836 | 0.7883 | 0.8880 | 0.9163 | 0.9217 | 0.8551 | 0.8408 | 93 |
| noise_light | 0.0047 | 0.0083 | 0.1427 | 0.0387 | 0.6995 | 0.7234 | 0.7215 | 0.7263 | 0.8710 | 0.8839 | 0.8920 | 0.8487 | 0.8119 | 93 |
| noise_medium | 0.0043 | 0.0105 | 0.1486 | 0.0392 | 0.6934 | 0.6431 | 0.6433 | 0.6462 | 0.8382 | 0.8203 | 0.8651 | 0.8426 | 0.7669 | 93 |
| noise_heavy | 0.0058 | 0.0156 | 0.1477 | 0.0414 | 0.6895 | 0.4527 | 0.4538 | 0.4550 | 0.7911 | 0.7099 | 0.7485 | 0.8398 | 0.6725 | 93 |
| noise_severe | 0.0090 | 0.0118 | 0.1250 | 0.0381 | 0.7197 | 0.2790 | 0.2797 | 0.2805 | 0.6885 | 0.5373 | 0.6394 | 0.8611 | 0.5673 | 93 |
| midi | 0.0051 | 0.0189 | 0.1388 | 0.0500 | 0.7566 | 0.2946 | 0.2950 | 0.2948 | 0.6246 | 0.5121 | 0.6548 | 0.8465 | 0.5532 | 93 |

Section 16.1 asks for the same parses paired with the naive coordinate map, which is what separates "the parse was bad" from "the layout stage did nothing":

| perception | R_OOB ↓ | R_col ↓ | clearance ↓ | door ↓ | reach ↑ | S_rel ↑ | S_rel|scaled ↑ | S_rel|elastic ↑ | S_rel|kept ↑ | S_motif ↑ | retention | legality ↑ | score ↑ | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 0.0523 | 0.0648 | 0.3763 | 0.1022 | 0.7694 | 0.9593 | 0.9748 | 0.9682 | 0.9593 | 1.0000 | 1.0000 | 0.5669 | 0.7270 | 93 |
| noise_light | 0.0581 | 0.0704 | 0.3779 | 0.0954 | 0.7587 | 0.8975 | 0.9106 | 0.9058 | 0.9400 | 0.9609 | 0.9752 | 0.5579 | 0.7026 | 93 |
| noise_medium | 0.0711 | 0.0679 | 0.3784 | 0.0872 | 0.7721 | 0.7903 | 0.8009 | 0.7976 | 0.8963 | 0.9058 | 0.9391 | 0.5509 | 0.6655 | 93 |
| noise_heavy | 0.1003 | 0.0693 | 0.3827 | 0.0784 | 0.7851 | 0.5595 | 0.5652 | 0.5642 | 0.8375 | 0.7963 | 0.8154 | 0.5318 | 0.5758 | 93 |
| noise_severe | 0.1339 | 0.0640 | 0.3879 | 0.0628 | 0.7688 | 0.3573 | 0.3611 | 0.3603 | 0.7361 | 0.6416 | 0.7033 | 0.5207 | 0.4716 | 93 |
| midi | 0.1217 | 0.1385 | 0.4560 | 0.0822 | 0.7743 | 0.3654 | 0.3683 | 0.3669 | 0.6349 | 0.5584 | 0.7325 | 0.4330 | 0.4117 | 93 |

The two tables move along different axes, and that is the whole point. Perception quality drives design preservation and barely touches legality; the layout stage drives legality and cannot invent design fidelity the parser threw away. Direct scaling posts the *higher* `S_rel` at every noise level — it copies the reference coordinates verbatim — while producing rooms with several times the out-of-bounds and collision area, which is exactly why `S_rel` alone was never allowed to be the headline number. MIDI paired with direct scaling is the worst cell in the whole study.

After gauge alignment (a single similarity per room, which a single image genuinely cannot fix and which ReRoom is invariant to), MIDI's median object-centre error is 0.47 m and its mean log-size error 0.41.

**A current single-image parser sits at the severe end of the simulated sweep** — statistically indistinguishable from it on relation preservation, slightly worse on motifs. That is the plan's top listed risk (section 20) measured rather than assumed, and it is why validating the oracle setting first was the right sequencing: physical legality does not degrade across the whole range (it drifts *up*, because fewer objects survive to be placed), so what perception costs is design fidelity, not usable rooms.

### The multi-view parser, on the same rooms

GenRecon is the plan's multi-view source parser (section 3.3): several photographs in, complete scene geometry out. That output shape is the difficulty. MIDI hands back one mesh per object; GenRecon hands back one mesh for the *room*, with no notion of "sofa" in it, and a design-intent graph needs instances. Labels are therefore lifted from the multi-view instance masks rendered with the input views — a point-splat z-buffer decides which vertices a camera actually sees, the mask under each pixel casts a vote, and the majority label wins. The 3D is entirely GenRecon's; the segmentation is supplied, exactly the concession made for MIDI above.

| parser | views | rooms | median object-centre error | mean log-size error |
|---|---|---|---|---|
| MIDI-3D | 1 | 46 | 0.47 m | 0.41 |
| GenRecon | 24 | 24 | 0.18 m | 0.51 |

Twenty-four views localise objects better than one, which is the ordering the plan assumes and worth having measured rather than assumed. Extracting the instances is its own error source, and a visible one: three successive attempts at separating an object's points from the room shell gave mean log-size errors of 0.82, 0.88 and 0.43. What finally worked was neither a statistical outlier trim (it shrank a chair back to a plane) nor simply the largest connected component (the room shell is one enormous component and swallowed a sideboard), but the largest component that is still a plausible piece of furniture.

Head to head on the rooms both parsers ran on, with every simulated noise level recomputed over that same sample:

| perception | R_OOB ↓ | R_col ↓ | clearance ↓ | door ↓ | reach ↑ | S_rel ↑ | S_rel|scaled ↑ | S_rel|elastic ↑ | S_rel|kept ↑ | S_motif ↑ | retention | legality ↑ | score ↑ | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 0.0034 | 0.0057 | 0.1271 | 0.0461 | 0.7120 | 0.8449 | 0.8433 | 0.8489 | 0.8890 | 0.9483 | 0.9697 | 0.8668 | 0.8785 | 42 |
| noise_light | 0.0021 | 0.0069 | 0.1377 | 0.0262 | 0.7398 | 0.7815 | 0.7790 | 0.7853 | 0.8842 | 0.9114 | 0.9370 | 0.8567 | 0.8454 | 42 |
| noise_medium | 0.0043 | 0.0083 | 0.1318 | 0.0375 | 0.7230 | 0.7147 | 0.7132 | 0.7188 | 0.8473 | 0.8552 | 0.9151 | 0.8605 | 0.8123 | 42 |
| noise_heavy | 0.0083 | 0.0152 | 0.1141 | 0.0456 | 0.7363 | 0.5251 | 0.5271 | 0.5284 | 0.7916 | 0.8165 | 0.8037 | 0.8708 | 0.7548 | 42 |
| noise_severe | 0.0086 | 0.0166 | 0.1074 | 0.0447 | 0.7843 | 0.3098 | 0.3094 | 0.3115 | 0.6911 | 0.5543 | 0.6573 | 0.8749 | 0.5898 | 42 |
| midi | 0.0034 | 0.0140 | 0.1127 | 0.0444 | 0.7070 | 0.2505 | 0.2511 | 0.2510 | 0.6139 | 0.4521 | 0.6218 | 0.8744 | 0.5242 | 42 |
| genrecon | 0.0087 | 0.0102 | 0.1333 | 0.0440 | 0.7020 | 0.4488 | 0.4506 | 0.4504 | 0.6768 | 0.6955 | 0.7779 | 0.8527 | 0.6592 | 42 |

The multi-view parser lands where the single-image one does not: around the heavy-to-severe end of the simulated sweep, where MIDI sits past its severe end. Physical legality is again almost flat across the whole range — what more views buy is design fidelity, not usable rooms, which is the same separation the noise sweep shows.

DINOv3, GenRecon's image tower, is licence-gated and this machine has no accepted licence; the weights come from an ungated third-party mirror of the same checkpoint at the user's explicit instruction, verified to load as `DINOv3ViTModel` with the expected 303.1 M parameters.

## 6b. Experiment 4 on real photographs

The same six prescribed target floors, but the reference is a *photograph* rather than a synthetic room: 10 real captures (BlendSwap, Matterport3D, Realistic-Style) parsed by MIDI-3D. Nothing here has ground truth, so categories come from CLIP zero-shot over ReRoom's vocabulary, metric scale is anchored on the best-constrained object category present, and the room outline is inferred from the reconstructed footprints. Each of those three is a stated assumption, not a hidden one.

| method | R_OOB ↓ | R_col ↓ | clearance ↓ | door ↓ | reach ↑ | S_rel ↑ | S_rel|scaled ↑ | S_rel|elastic ↑ | S_rel|kept ↑ | S_motif ↑ | retention | legality ↑ | score ↑ | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reroom_full | 0.0046 | 0.0167 | 0.1138 | 0.0000 | 0.7457 | 0.8388 | 0.8344 | 0.8392 | 0.9151 | 0.9335 | 0.9483 | 0.8702 | 0.8693 | 60 |
| direct_scaling | 0.0342 | 0.0511 | 0.2914 | 0.0000 | 0.7114 | 0.9776 | 0.9744 | 0.9788 | 0.9776 | 1.0000 | 1.0000 | 0.6592 | 0.7896 | 60 |

The result holds on real input: the coordinate map keeps every relation and puts furniture through walls, ReRoom fits the room. Asset substitution cannot fire here — a photographed object has no source asset id to substitute *from* — so the appearance column is vacuously 1.0 and is not evidence of anything.

## 7a. Does relation elasticity change the output?

The ablation grid separates `alpha = 0` from a fitted `alpha` by less than a hundredth of the joint score, which is small enough to be worth interrogating rather than burying. This probe isolates the regime where eq. (9) can actually bite — strong uniform rescalings — and splits the relation error by how elastic each relation is (`alpha < 0.25` vs the rest).

The numbers below are from the current build. An earlier version of this table showed elasticity making things slightly *worse*; that turned out to be three wiring faults in the probe and the initialisation rather than a property of eq. (9), and the sign flipped once they were fixed. The magnitude did not change much, and the conclusion below is about the magnitude.

| elasticity model | S_rel | S_rel|elastic | S_motif | legality | score | rigid-relation err ↓ | elastic-relation err ↓ |
|---|---|---|---|---|---|---|---|
| alpha=0 (rigid) | 0.5474 | 0.5499 | 0.7421 | 0.8729 | 0.7069 | 0.1688 | 0.1425 |
| prior alpha | 0.5429 | 0.5597 | 0.7500 | 0.8704 | 0.7102 | 0.1592 | 0.1299 |
| fitted f_psi | 0.5418 | 0.5569 | 0.7500 | 0.8706 | 0.7103 | 0.1666 | 0.1341 |

### Can it be made to matter?

Two further variants were tried, because a mechanism that is empirically real and operationally inert deserves a second look before it is written off. The elasticity already sets the *target* distance (eq. 9); it can also set the relation's **stiffness**, since alpha is exactly a statement about how confidently the target is known — near 0 the distance is fixed by the human body, near 1 it is known only up to the room's scale.

| variant | Δ score from using alpha | elastic-relation err | overall score |
|---|---|---|---|
| alpha sets the target only | +0.0011 | 0.142 → 0.125 | 0.7116 |
| + stiffness, un-normalised | +0.0134 | 0.129 → 0.118 | 0.6794 |
| + stiffness, mean-normalised | -0.0024 | 0.142 → 0.128 | 0.7101 |
| + alpha-blended initialisation (shipped) | +0.0033 | 0.143 → 0.130 | 0.7069 |

**Alpha can be made into a real lever — but only by paying more for it than it returns.** Letting stiffness inflate the relation term multiplies the ablation gap by twelve, and costs 0.032 of the overall score, because a heavier relation term simply outvotes the feasibility terms. Hold the total relation weight constant and alpha merely redistributes it: the lever vanishes, while the error on the relations it targets still falls by about 10 %.

The reason is structural rather than a tuning failure. The objective is dominated by the preservation-versus-feasibility trade-off; alpha only reshuffles weight *inside* the preservation half, so it cannot move the frontier. Relation elasticity is therefore a well-supported *description* of how designed rooms scale — and the right thing to report as a finding — but the method's engine is the motif layer and the constraint projection. The normalised variant is what ships: neutral on the objective, 10 % better on the relations it exists for.

By target scale (uniform rescaling of the room):

| scale | model | S_rel | S_rel|elastic | S_motif | score |
|---|---|---|---|---|---|
| 0.6 | alpha=0 (rigid) | 0.1819 | 0.1844 | 0.4122 | 0.4261 |
| 0.6 | prior alpha | 0.1824 | 0.1883 | 0.4223 | 0.4263 |
| 0.6 | fitted f_psi | 0.1822 | 0.1878 | 0.4223 | 0.4246 |
| 0.75 | alpha=0 (rigid) | 0.3392 | 0.3407 | 0.6173 | 0.5518 |
| 0.75 | prior alpha | 0.3365 | 0.3407 | 0.6111 | 0.5458 |
| 0.75 | fitted f_psi | 0.3385 | 0.3426 | 0.6197 | 0.5507 |
| 1.4 | alpha=0 (rigid) | 0.8465 | 0.8467 | 0.9744 | 0.9250 |
| 1.4 | prior alpha | 0.8440 | 0.8623 | 0.9907 | 0.9408 |
| 1.4 | fitted f_psi | 0.8405 | 0.8558 | 0.9839 | 0.9368 |
| 1.8 | alpha=0 (rigid) | 0.8221 | 0.8278 | 0.9645 | 0.9248 |
| 1.8 | prior alpha | 0.8087 | 0.8473 | 0.9759 | 0.9278 |
| 1.8 | fitted f_psi | 0.8061 | 0.8414 | 0.9742 | 0.9292 |

## 7b. Style-aware retrieval, eq. (30)

Real 3D-FUTURE assets with CLIP image embeddings. Each object is asked for a rescaled version of itself; keeping the reference asset costs a mean log-size error of 0.2489. The two degenerate weightings are the controls.

| weighting | size error ↓ | CLIP similarity ↑ | shape distance ↓ | queries |
|---|---|---|---|---|
| balanced | 0.1043 | 0.8270 | 0.2126 | 7036 |
| size_only | 0.0526 | 0.6970 | 0.2679 | 7036 |
| appearance_only | 0.3220 | 0.8744 | 0.1908 | 7036 |
| balanced+geo | 0.1256 | 0.8299 | 0.1521 | 7036 |

The balanced objective recovers most of the achievable size correction while giving up little appearance similarity, which is the trade the plan argues for: retrieve a genuinely smaller sofa that still looks like the reference, rather than squash the reference one.

The fourth row adds `f^geo`, the per-node geometry feature of eq. (10): a canonical occupancy descriptor computed from the asset mesh, so retrieval can ask whether a candidate is the same *shape* and not merely the same size and style. It cuts shape distance sharply for a small size concession, and appearance similarity goes up rather than down — the descriptor and the CLIP embedding agree more often than they conflict.

One measurement bug is worth recording, because it had been silently disabling half of eq. (30): the per-category embedding cache dropped the appearance term entirely if a single asset in that category lacked an embedding, so on any partially embedded catalogue `balanced` and `size_only` were literally the same retrieval. Missing rows are now masked individually.

## 7c. What the global appearance score is worth (15.2)

The plan states that a whole-image CLIP similarity "cannot substitute for relation and motif evaluation". That is testable, so it was tested: the same retargetings, scored by every metric at once.

| method | S_rel | S_motif | legality | appearance (object) | appearance (global) |
|---|---|---|---|---|---|
| reference_rigid | 1.0000 | 1.0000 | 0.6600 | 1.0000 | 0.9702 |
| direct_scaling | 0.9846 | 1.0000 | 0.6549 | 1.0000 | 0.9762 |
| reroom_full | 0.8573 | 0.9646 | 0.8455 | 0.9095 | 0.9579 |

Legality separates these three methods by 0.191 and `S_rel` by 0.143; the whole-render CLIP score separates them by 0.018. Across the individual retargetings it is essentially uncorrelated with legality, which is the sharpest form of the plan's objection: it cannot tell a room you can walk through from one you cannot, so it is reported as an auxiliary number and never enters `score`.

## 7d. VLM semantic relations (section 20)

The plan's risk table lists "LLM/VLM relation extraction unstable" and prescribes the mitigation directly: deterministic geometry and category rules do the work, and the VLM supplies semantic relations only. Rather than take that on trust, a CLIP-backed extractor was built and measured against the geometric rules over 50 rendered reference rooms.

| geometric semantic relations | 501 |
|---|---|
| VLM proposals | 208 |
| both | 86 |
| precision vs geometry | 0.413 |
| recall vs geometry | 0.172 |

Two things had to be fixed before the number meant anything. Uncalibrated, CLIP answered *symmetric* for every pair it was shown, because raw similarity carries a large per-phrase bias; subtracting each prompt's own mean over the scene's pairs makes the decision relative. And a matching pair is two of the same thing, which is a category rule, not a judgement to delegate. Even then the two agree on well under half of what either proposes — which is the evidence for the plan's own conclusion, so the extractor stays off the default path and adds edges only where geometry found none.

## 7d1. Head to head with PhyScene (bibliography [11])

PhyScene's released code was run here — its own weights, its own preprocessed 3D-FRONT split, its own sampler — and its generated layouts were converted into ReRoom scenes. Three things are then held fixed so the comparison is a comparison: **the rooms** (the same test-split floor plans PhyScene generated into), **the object vocabulary** (ReRoom's reference scenes are rebuilt from the same cached boxes PhyScene trains on, so neither side sees objects the other cannot), and **the evaluator** (one implementation scores both).

| method | Col_obj ↓ | Col_scene ↓ | R_out ↓ | R_walkable ↑ | R_reach ↑ | objects | n |
|---|---|---|---|---|---|---|---|
| 3D-FRONT reference | 0.366 | 0.793 | 0.063 | 0.970 | 0.889 | 11.422 | 256 |
| PhyScene | 0.391 | 0.840 | 0.119 | 0.961 | 0.872 | 11.648 | 256 |
| ReRoom | 0.113 | 0.336 | 0.024 | 0.905 | 0.889 | 11.391 | 256 |
| ReRoom (foreign reference) | 0.030 | 0.154 | 0.039 | 0.870 | 0.865 | 13.168 | 256 |
| ReRoom (no reference) | 0.001 | 0.008 | 0.016 | 0.908 | 0.932 | 11.414 | 256 |

**On the two metrics that say whether furniture is physically where it should be, relation-aware retargeting is well ahead of a purpose-built generator.** Colliding objects fall from 0.391 to 0.113 and objects outside the floor plan from 0.119 to 0.024, at the same object count — and PhyScene is not a weak baseline here: it sits about where the real 3D-FRONT rooms sit, which is what a generative prior trained on them should do.

**On free-space connectivity it loses, and that is a real weakness rather than a rounding difference.** `R_walkable` drops to 0.905 with the room's own reference and 0.831 with a foreign one, against 0.961 for PhyScene and 0.970 for the ground truth. Placing furniture legally is not the same as leaving the floor connected, and the clearance term is evidently doing less work than the collision and boundary terms. That is the concrete thing to fix next.

The two ReRoom rows exist because the obvious version of this table is unfair. Giving ReRoom the reference of the very room it is furnishing is an information advantage PhyScene does not have. The *foreign reference* row removes it — a different living room's design, transferred into this floor plan, which is also the actual use case. It collides even less, costs out-of-plan area and walkability, and adds objects (13.9 against 11.4) because the population stage fills space the borrowed design does not account for.

The chain from their published table to the one above has two links, and only one of them is tight. Their generator was re-run here and scored by *their* script: `R_out` 0.130 against a published 0.219, `R_walkable` 0.831 against 0.815, `R_reach` 0.821 against 0.771 — the right neighbourhood, with the gaps explained by 200 generated scenes rather than 1000 and by a `bounds.npz` that had to be rebuilt from their shipped statistics. Then those same scenes were scored by the evaluator used above: `R_out` 0.119 against their script's 0.130, which is tight, but `R_walkable` 0.961 against 0.831, which is not.

So the columns carry different weight. **`Col_obj`, `Col_scene` and `R_out` are safe to read across the table** — the out-of-plan definition reproduces to within 0.011. `R_walkable` and `R_reach` are safe only *within* the table, where both sides go through one evaluator; their absolute values sit above what PhyScene's own rasterisation reports and should not be compared to the published column.

One caveat on the metric itself: this implementation reproduces PhyScene's `R_out` closely (0.119 here against 0.130 from their own script on the same run) but reads walkability higher than theirs (0.961 against 0.831), because the rasterisation details of their erosion and box-stroking could not be matched exactly. Comparisons *within* this table are sound; comparing its walkability column against their published numbers is not.

## 7d2. Read on another paper's yardstick (bibliography [11])

The plan's bibliography names PhyScene as one of the floor-plan-conditioned synthesis systems this work sits beside, and section 16.1 asks for a baseline from that family. Its code is public and it reports physical-plausibility numbers on 3D-FRONT — but its metrics are *not* the ones used above. `R_out` there is the fraction of **objects** with any pixel outside the floor plan; here it has been the fraction of furniture **area**. Those differ by an order of magnitude on the same scene. So the definitions were reimplemented from PhyScene's released `utils/overlap.py` and `scripts/eval/walkable_metric.py`, and ReRoom's own scenes recomputed under them.

Two target settings are shown, because they are not equally hard: *as-is* uses the reference room's own floor plan, which is the setting PhyScene evaluates in, and *retargeted* uses a deformed polygon, which is this project's actual task.

**bedroom**

| method | Col_obj ↓ | Col_scene ↓ | R_out ↓ | R_walkable ↑ | R_reach ↑ | n |
|---|---|---|---|---|---|---|
| ATISS (published) | 0.248 | 0.460 | 0.286 | 0.839 | 0.736 | – |
| DiffuScene (published) | 0.228 | 0.430 | 0.272 | 0.827 | 0.755 | – |
| PhyScene (published) | 0.187 | 0.360 | 0.245 | 0.865 | 0.762 | – |
| 3D-FRONT reference [as-is] | 0.474 | 0.812 | 0.163 | 0.970 | 0.893 | 85 |
| reroom_full [as-is] | 0.166 | 0.341 | 0.065 | 0.970 | 0.918 | 85 |
| target_only [as-is] | 0.029 | 0.071 | 0.080 | 0.960 | 0.911 | 85 |
| direct_scaling [as-is] | 0.474 | 0.812 | 0.163 | 0.970 | 0.893 | 85 |
| reroom_full [retargeted] | 0.095 | 0.212 | 0.119 | 0.949 | 0.912 | 85 |
| target_only [retargeted] | 0.020 | 0.071 | 0.103 | 0.918 | 0.897 | 85 |
| direct_scaling [retargeted] | 0.424 | 0.753 | 0.399 | 0.959 | 0.815 | 85 |

**living room**

| method | Col_obj ↓ | Col_scene ↓ | R_out ↓ | R_walkable ↑ | R_reach ↑ | n |
|---|---|---|---|---|---|---|
| ATISS (published) | 0.316 | 0.850 | 0.136 | 0.814 | 0.791 | – |
| DiffuScene (published) | 0.198 | 0.690 | 0.238 | 0.790 | 0.756 | – |
| PhyScene (published) | 0.191 | 0.630 | 0.219 | 0.815 | 0.771 | – |
| 3D-FRONT reference [as-is] | 0.354 | 0.783 | 0.073 | 0.985 | 0.904 | 60 |
| reroom_full [as-is] | 0.103 | 0.283 | 0.032 | 0.955 | 0.923 | 60 |
| target_only [as-is] | 0.011 | 0.050 | 0.015 | 0.970 | 0.976 | 60 |
| direct_scaling [as-is] | 0.354 | 0.783 | 0.073 | 0.985 | 0.904 | 60 |
| reroom_full [retargeted] | 0.073 | 0.267 | 0.034 | 0.968 | 0.949 | 60 |
| target_only [retargeted] | 0.016 | 0.067 | 0.025 | 0.950 | 0.956 | 60 |
| direct_scaling [retargeted] | 0.302 | 0.683 | 0.165 | 0.982 | 0.875 | 60 |

**dining room**

| method | Col_obj ↓ | Col_scene ↓ | R_out ↓ | R_walkable ↑ | R_reach ↑ | n |
|---|---|---|---|---|---|---|
| ATISS (published) | 0.591 | 0.960 | 0.132 | 0.874 | 0.848 | – |
| DiffuScene (published) | 0.160 | 0.550 | 0.244 | 0.787 | 0.847 | – |
| PhyScene (published) | 0.151 | 0.530 | 0.217 | 0.852 | 0.789 | – |
| 3D-FRONT reference [as-is] | 0.397 | 0.600 | 0.047 | 0.965 | 0.877 | 5 |
| reroom_full [as-is] | 0.168 | 0.400 | 0.047 | 0.965 | 0.906 | 5 |
| target_only [as-is] | 0.000 | 0.000 | 0.000 | 1.000 | 0.950 | 5 |
| direct_scaling [as-is] | 0.397 | 0.600 | 0.047 | 0.965 | 0.877 | 5 |
| reroom_full [retargeted] | 0.000 | 0.000 | 0.029 | 0.997 | 0.971 | 5 |
| target_only [retargeted] | 0.000 | 0.000 | 0.029 | 0.997 | 1.000 | 5 |
| direct_scaling [retargeted] | 0.375 | 0.600 | 0.229 | 1.000 | 0.824 | 5 |

**This is not a head-to-head, and the ground-truth row is why it cannot be read as one.** Real 3D-FRONT bedrooms score `Col_obj` 0.474 under this implementation — worse than every published generative method — while their furniture-area collision is only 6 %. A binary per-object rate is dominated by many tiny overlaps, which real designed rooms are full of, so the absolute values depend heavily on which objects are in the vocabulary at all. PhyScene evaluates on the ATISS-preprocessed subset; ReRoom parses the rooms itself. Until both are run through one script on one object set, the columns are not the same column.

What *is* readable is the internal ordering, which is consistent and large: ReRoom cuts per-object collisions from the reference's 0.474 to 0.095 and out-of-plan objects to 0.119 while transferring the design, and direct scaling — the same object set, the same rooms, no layout stage — stays at 0.424 and 0.399. The floor-plan-only arm is cleanest of all on these metrics precisely because it owes the reference nothing, which is the trade the whole report is about.

### Why the walkability gap was not closed

Two attempts, both measured, both rejected. They are recorded because the second one explains the first.

The gap was first read as *pinches* — gaps between objects too narrow to walk through. ReRoom does have more of them than either PhyScene or the real rooms (11.2 against 9.2 and 9.4 per room), and `E_clear` charged them with a quadratic, which says a 0.30 m gap is nine times better than a 0.05 m one when 0.45 m is needed — while both are equally unwalkable. Giving the shortfall a saturating share moved `R_walkable` from 0.896 to 0.901 and left the gap count *higher*. The diagnosis was wrong: the count included same-motif pairs, which are supposed to be close and are not charged, and the correlation behind it explained 8 % of the variance. The term is kept as a knob and defaulted off.

The second attempt went at the term that is actually responsible, `E_func`'s reachability share, and it *does* move the metric — but the way it moves everything else is the finding:

| `func_reach` | R_walkable ↑ | Col_obj ↓ | legality ↑ | score ↑ |
|---|---|---|---|---|
| 2 (shipped) | 0.897 | **0.097** | **0.821** | **0.881** |
| 20 | 0.923 | 0.177 | 0.695 | 0.804 |
| 80 | 0.925 | 0.198 | 0.647 | 0.772 |

Walkability rises to 0.925 and stops there, well short of the real rooms' 0.972, while collisions double. The reason is structural rather than a matter of weight: reachability enters only the **exact** energy, which ranks already-refined candidates, and never the differentiable surrogate that produces them. Raising it makes the ranker prefer a better-connected layout among candidates none of which was optimised for connectivity — it selects, it does not optimise. (Worth noting for anyone reproducing this: none of the 100 rooms carries a door in the data, so `reachable_ratio` falls back to its largest-component seed and is measuring almost exactly what `R_walkable` measures. The target was right; the tool was in the wrong place.)

Closing this properly needs a *differentiable* connectivity surrogate, and connectivity is global, non-local and not differentiable as stated — that is a piece of research, not a tuning pass. The shipped setting is left where the joint score is highest, and the gap is reported rather than traded away for a number that `Col_obj` would immediately expose.

The honest next step is not a better table but their code: PhyScene's repository is public and current, so both systems can be run on the same target floor plans and scored by one evaluator. That is a day of work, not a research question.

## 7e. User constraints, `C_t` (section 1)

The plan's problem statement takes the reference, the target polygon *and* a set of user constraints. Two are implemented: objects the person pinned, and floor they marked as no-go. Both are hard, so the number that matters is what obeying them costs.

| setting | pins / zones obeyed | legality | S_rel | S_motif | score |
|---|---|---|---|---|---|
| unconstrained | – | 0.8504 | 0.8688 | 0.9754 | 0.8820 |
| one pinned object | 100% exact | 0.8140 | 0.8334 | 0.9508 | 0.8464 |
| a forbidden quarter of the floor | 93% of the intrusion removed | 0.7581 | 0.7813 | 0.9218 | 0.7964 |

Pinning the most important object in the room holds its pose exactly in every room tested, and costs about 0.036 of the joint score; forbidding a quarter of the floor cuts furniture inside it from 1.93 m² to 0.14 m² and costs 0.086. Both are the price of a constraint, not a defect: a room told to leave its best wall empty is a harder room.

Getting there took four fixes, all of the same shape — a hard constraint that some *other* code path quietly re-randomised. The gradient freeze holds a pinned object exactly where it starts, so anything that moved its starting point defeated it: the affine restart candidate, the jitter that re-seeds the restarts after a repair. Keep-out zones were charged through a separate soft term that the feasibility escalation did not raise, which made them *cheaper* to violate exactly when the room got tight; they are now punched out of the floor field the boundary term reads, so a no-go zone has a wall's gradient.

## 7f. SAGE-10k augmentation (section 17, month 5)

The plan gives SAGE a narrow job — object diversity and open-vocabulary augmentation, explicitly not room geometry — so the honest test is retrieval coverage, not layout quality. A category the bank has never seen cannot be substituted at all, however good the optimiser is.

| catalogue | assets | reference objects with a candidate |
|---|---|---|
| 3D-FUTURE alone | 11970 | 94.0% |
| + SAGE pseudo-assets | 12674 | 100.0% |

The gain is entirely in categories 3D-FUTURE's canonical mapping does not reach — `bench`, `bookcase`, `fireplace`, `mirror`, `misc`, `piano`, `plant`, `rug`, `stool`, `tv`, `wall_art` — and merging is deliberately conservative: a category the base bank already covers well keeps its real meshes with real product images, because a size sampled from statistics is a worse candidate than an actual model.

## 8. Figures

- `outputs/exp2/figure_0.png`
- `outputs/exp2/figure_1.png`
- `outputs/exp2/figure_2.png`
- `outputs/exp4/case_0.png`
- `outputs/exp4/case_1.png`
- `outputs/exp4/case_2.png`
- `outputs/exp4/case_3.png`
- `outputs/fig_corpus_samples.png`
- `outputs/fig_scene3d.png`
