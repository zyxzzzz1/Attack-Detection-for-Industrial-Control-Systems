# Cases

> Asset mapping and vulnerability data come from a lab environment and are for demonstration only. The asset mapping is constructed by hand and the vulnerability data comes from scanning a self-built target machine; neither has any causal relation to the attacks in HAI. The intrusion paths and vulnerabilities on the cards only show how the prototype brings the information together; they are not a conclusion about where these attacks came from.

The alerts are all 1703 alarm segments of the LSTM-AE main model run (lstm_ae, seed 42) on the HAI 21.03 test set. The cases are selected by a fixed rule, not by hand: going down the fusion priority ranking, take the first real attack with correct attribution, the first real attack with wrong attribution, and the first false alarm. "Whether it is a real attack and whether the attribution is a hit" comes from an evaluation done afterwards with the labels; the cards do not contain this information.

## Case 1 — a real attack with correct attribution: LSTMAE-00112

- Fusion priority rank: 1 / 1703 (priority 2.000, S 1.000, E_sub 1.000)
- Alert time: 2020-07-09 15:47:47 ~ 2020-07-09 15:50:12 (146 s, test2)
- Max anomaly score 1592, i.e. 6721.3 times the threshold (0.2369)
- Top-5 sensors (share of total sensor error): P2_VT01 33.3%, P2_SIT02 25.0%, P2_SIT01 25.0%, P2_CO_rpm 13.2%, P2_VYT03 1.3%
- Subsystems inferred by the card: P2 100% (criticality 1.0, E_sub 1)
- Intrusion paths given by the card: ews_01 → plc_p2; historian → hmi_main → plc_p2
- Assets on the path that have not been assessed: historian, hmi_main, plc_p2; 10 vulnerabilities listed (the top one is Apache Tomcat AJP RCE Vulnerability (Ghostcat) - Active Check on ews_01), 7 needing manual verification
- Alert card: [`cases/case1_LSTMAE-00112.md`](cases/case1_LSTMAE-00112.md)

**Against the labels**: this alert overlaps attack event #6 (2020-07-09 15:47:00 ~ 2020-07-09 15:48:22, 83 s, attacked subsystem according to the labels P2; detection delay of this event 47 s). The Top-1 sensor P2_VT01 belongs to P2, which **is** in the event's subsystem set: the attribution is a hit.

## Case 2 — a real attack with wrong attribution: LSTMAE-00472

- Fusion priority rank: 50 / 1703 (priority 1.412, S 1.000, E_sub 0.677)
- Alert time: 2020-07-28 12:43:22 ~ 2020-07-28 12:47:18 (237 s, test4)
- Max anomaly score 5.488, i.e. 23.2 times the threshold (0.2369)
- Top-5 sensors (share of total sensor error): P4_ST_PT01 24.8%, P1_PIT01 22.0%, P1_FT01 11.5%, P1_FT01Z 10.9%, P1_LCV01Z 7.5%
- Subsystems inferred by the card: P1 68% (criticality 0.9, E_sub 1), P4 32% (criticality 0.6, E_sub 0)
- Intrusion paths given by the card: ews_01 → plc_p1; historian → hmi_main → plc_p1
- Assets on the path that have not been assessed: historian, hmi_main, plc_p1; 10 vulnerabilities listed (the top one is Apache Tomcat AJP RCE Vulnerability (Ghostcat) - Active Check on ews_01), 7 needing manual verification
- Alert card: [`cases/case2_LSTMAE-00472.md`](cases/case2_LSTMAE-00472.md)

**Against the labels**: this alert overlaps attack event #34 (2020-07-28 12:43:03 ~ 2020-07-28 12:47:16, 254 s, attacked subsystem according to the labels P1; detection delay of this event 19 s). The Top-1 sensor P4_ST_PT01 belongs to P4, which is **not** in the event's subsystem set: by the rule of Decision 7 the attribution is a miss.

## Case 3 — a high-priority false alarm: LSTMAE-01505

- Fusion priority rank: 28 / 1703 (priority 1.598, S 1.000, E_sub 0.788)
- Alert time: 2020-07-31 08:41:25 ~ 2020-07-31 08:42:04 (40 s, test5)
- Max anomaly score 0.5164, i.e. 2.2 times the threshold (0.2369)
- Top-5 sensors (share of total sensor error): P1_PIT01 14.0%, P1_FT01Z 12.5%, P4_ST_PT01 10.6%, P2_HILout 6.5%, P2_VT01 6.4%
- Subsystems inferred by the card: P1 53% (criticality 0.9, E_sub 1), P2 26% (criticality 1.0, E_sub 1), P4 21% (criticality 0.6, E_sub 0)
- Intrusion paths given by the card: ews_01 → plc_p1; historian → hmi_main → plc_p1; ews_01 → plc_p2; historian → hmi_main → plc_p2
- Assets on the path that have not been assessed: historian, hmi_main, plc_p1, plc_p2; 10 vulnerabilities listed (the top one is Apache Tomcat AJP RCE Vulnerability (Ghostcat) - Active Check on ews_01), 7 needing manual verification
- Alert card: [`cases/case3_LSTMAE-01505.md`](cases/case3_LSTMAE-01505.md)

**Against the labels**: this alert does not overlap any attack event; it is a false alarm. In the same file, the previous attack event #47 ended at 2020-07-31 02:05:11 (396 minutes earlier), the next attack event #48 started at 2020-07-31 09:54:01 (72 minutes later).

## Taken together

- In Case 2 the Top-1 sensor is not in the attacked subsystem, but most of the other Top-5 sensors are: after weighting by the Top-5 error shares the card still puts 68% of the weight on P1, and the intrusion paths it gives belong to that subsystem. In other words, "whether Top-1 is a hit" is stricter than the information the card actually shows; this holds for this one alert only and must not be generalised.
- In Case 2 the Top-1 sensor falls in P4 (the HIL simulation part, which has no subsystem label, so by the rule it always counts as a miss). P4 is physically coupled with P1, P2 and P3, so pointing to P4 is not necessarily wrong; the available labels cannot tell. The card carries a note on P4 and recommends investigating P1 / P2 / P3 as well.
- The false alarm of Case 3 (score 2.2 times the threshold) is ranked ahead of the real attack of Case 2 (23.2 times). The reason is that the anomaly severity S = min(1, score ÷ threshold − 1) is capped at 1 as soon as the score reaches 2 times the threshold, so both alerts have the same S and the ranking is decided only by subsystem weights, criticality and E_sub. The cap on S is too low; this is a flaw of the scoring formula.
- The vulnerabilities listed on all three cards come from the same scanned asset (ews_01), and the top entry is the same one: only a single target machine has been assessed, so every alert that involves P1 or P2 receives almost the same vulnerability context, with no difference between real attacks and false alarms. This is a limitation caused by the single target machine and by risk saturation (see the M9 section of PROGRESS.md).
- The card of the false alarm looks exactly like the cards of the real attacks (the same intrusion paths, vulnerabilities and remediation advice). A card does not help decide whether an alert is a false alarm; it only gives a starting point for investigation on the assumption that the alert is real.
