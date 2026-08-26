#!/usr/bin/env python3
"""Publish a curated batch of real Mission2 test-split windows through the
live pipeline, for a demo incident strip that isn't almost entirely
pre-gate/no-decision incidents.

Every window below is a REAL row from the test split, not invented -- same
convention as scripts/publish_stub.py, just more of them, and picked (via a
one-off analysis using the committed DetectorArtifact and
ReferenceContextProvider, not guessed) to span the gate's distance range so
the resulting incidents cover confirm, disputed, and reject rather than
clustering on one verdict:

    label       channel     window_start          distance  verdict   severity   destination
    anomaly     channel_18  2001-12-14 19:00:00   0.91      reject    Advisory   log
    anomaly     channel_18  2001-12-14 20:00:00   1.47      disputed  Advisory   log
    rare_event  channel_20  2001-12-31 16:00:00   12.45     confirm   Critical   flight_director
    rare_event  channel_21  2002-07-31 03:00:00   8.07      confirm   Caution    subsystem_engineer
    rare_event  channel_20  2002-12-31 14:00:00   1.91      disputed  Advisory   log
    rare_event  channel_23  2002-05-27 07:00:00   0.57      reject    Advisory   log
    rare_event  channel_21  2003-01-05 19:00:00   0.57      reject    Advisory   log
    nominal     channel_23  2002-04-25 22:00:00   0.52      reject    Advisory   log
    nominal     channel_24  2003-06-25 03:00:00   0.53      reject    Advisory   log
    nominal     channel_24  2002-05-06 12:00:00   0.54      reject    Advisory   log

Distances above are pre-publish estimates from that analysis; the actual
incident is scored live by the real pipeline (detector-service, then
triage-service's gate), so treat this table as "why these ten", not as the
recorded outcome -- read that back from Firestore after publishing.

    python3 scripts/publish_demo_batch.py
    python3 scripts/publish_demo_batch.py --delay 5   # seconds between publishes
"""
import argparse
import json
import os
import time

from google.cloud import pubsub_v1

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "aksha-hackathon")
TOPIC = "telemetry-in"

WINDOWS = [
    {
        "channel_id": "channel_18", "label": "anomaly", "window_start": "2001-12-14 19:00:00",
        "features": {
            "sample_count": 240.0, "total_gap_seconds": 0.0, "max_gap_seconds": 18.003, "gap_fraction": 0.0,
            "mean": 0.4560243545472622, "std": 0.0003144146000561613, "var": 9.885654072847585e-08,
            "skew": 0.004579935174022468, "kurtosis": 2.5920010249597594, "min": 0.45473334193229675,
            "max": 0.45708829164505005, "mean_abs_change": 0.00025230541301133046, "n_peaks": 51.0,
            "smooth_n_peaks": 38.0, "diff_peaks": 58.0, "diff_var": 1.590809063999277e-07, "diff2_peaks": 62.0,
            "diff2_var": 3.711643816928215e-07, "slope": -7.078895366505782e-08, "mahalanobis": 1.032868566509986,
            "seconds_since_last_tc": 576.332, "tc_count_in_window": 53.0,
        },
    },
    {
        "channel_id": "channel_18", "label": "anomaly", "window_start": "2001-12-14 20:00:00",
        "features": {
            "sample_count": 240.0, "total_gap_seconds": 0.0, "max_gap_seconds": 18.003, "gap_fraction": 0.0,
            "mean": 0.45602683410048483, "std": 0.00010427214904123121, "var": 1.0872681065676734e-08,
            "skew": 0.0447953367287284, "kurtosis": 0.4876969548625967, "min": 0.4557673931121826,
            "max": 0.45638635754585266, "mean_abs_change": 8.891500420306795e-05, "n_peaks": 45.0,
            "smooth_n_peaks": 34.0, "diff_peaks": 52.0, "diff_var": 1.2682961248890056e-08, "diff2_peaks": 56.0,
            "diff2_var": 2.150873514536102e-08, "slope": -2.145574984595948e-09, "mahalanobis": 0.9625202533779513,
            "seconds_since_last_tc": 576.325, "tc_count_in_window": 59.0,
        },
    },
    {
        "channel_id": "channel_20", "label": "rare_event", "window_start": "2001-12-31 16:00:00",
        "features": {
            "sample_count": 240.0, "total_gap_seconds": 0.0, "max_gap_seconds": 18.003, "gap_fraction": 0.0,
            "mean": 0.4558077727258205, "std": 0.009535362396597325, "var": 9.092313603444227e-05,
            "skew": -2.3655165659745814, "kurtosis": 17.39091309173851, "min": 0.3867502510547638,
            "max": 0.4881495535373688, "mean_abs_change": 0.00745789249937738, "n_peaks": 69.0,
            "smooth_n_peaks": 58.0, "diff_peaks": 71.0, "diff_var": 0.0002655083476410575, "diff2_peaks": 78.0,
            "diff2_var": 0.0010168827634263065, "slope": -3.9865450384151105e-06, "mahalanobis": 4.133187765219471,
            "seconds_since_last_tc": 573.853, "tc_count_in_window": 55.0,
        },
    },
    {
        "channel_id": "channel_21", "label": "rare_event", "window_start": "2002-07-31 03:00:00",
        "features": {
            "sample_count": 240.0, "total_gap_seconds": 0.0, "max_gap_seconds": 18.003, "gap_fraction": 0.0,
            "mean": 0.16040713638067244, "std": 0.002925633941997496, "var": 8.559333962567807e-06,
            "skew": 0.4638903259090157, "kurtosis": 1.0256267649868924, "min": 0.15637513995170593,
            "max": 0.16960124671459198, "mean_abs_change": 0.00012041238983671869, "n_peaks": 10.0,
            "smooth_n_peaks": 6.0, "diff_peaks": 29.0, "diff_var": 5.462872727306999e-07, "diff2_peaks": 49.0,
            "diff2_var": 1.0993576298168103e-06, "slope": 2.3800683854839265e-05, "mahalanobis": 1.7660553677522406,
            "seconds_since_last_tc": 542.649, "tc_count_in_window": 67.0,
        },
    },
    {
        "channel_id": "channel_20", "label": "rare_event", "window_start": "2002-12-31 14:00:00",
        "features": {
            "sample_count": 240.0, "total_gap_seconds": 0.0, "max_gap_seconds": 18.003, "gap_fraction": 0.0,
            "mean": 0.4562892563641071, "std": 0.00020114630536218026, "var": 4.045983616085547e-08,
            "skew": -0.16616899978552635, "kurtosis": 0.34464399546823987, "min": 0.45553314685821533,
            "max": 0.45684680342674255, "mean_abs_change": 0.00020710098084492898, "n_peaks": 69.0,
            "smooth_n_peaks": 47.0, "diff_peaks": 84.0, "diff_var": 6.471591708904645e-08, "diff2_peaks": 86.0,
            "diff2_var": 1.9297925773674632e-07, "slope": 1.7850460083768925e-07, "mahalanobis": 17.192128716836887,
            "seconds_since_last_tc": 520.131, "tc_count_in_window": 58.0,
        },
    },
    {
        "channel_id": "channel_23", "label": "rare_event", "window_start": "2002-05-27 07:00:00",
        "features": {
            "sample_count": 240.0, "total_gap_seconds": 0.0, "max_gap_seconds": 18.003, "gap_fraction": 0.0,
            "mean": 0.8803099855780602, "std": 9.368565223413094e-05, "var": 8.777001434534524e-09,
            "skew": 0.7144121712817102, "kurtosis": 0.3462008130370977, "min": 0.880125880241394,
            "max": 0.8805806040763855, "mean_abs_change": 1.2219551220611112e-05, "n_peaks": 13.0,
            "smooth_n_peaks": 7.0, "diff_peaks": 25.0, "diff_var": 1.2055263307477885e-09, "diff2_peaks": 51.0,
            "diff2_var": 2.4232922559730643e-09, "slope": 4.886917008203417e-07, "mahalanobis": 1.0681129045348075,
            "seconds_since_last_tc": 552.181, "tc_count_in_window": 62.0,
        },
    },
    {
        "channel_id": "channel_21", "label": "rare_event", "window_start": "2003-01-05 19:00:00",
        "features": {
            "sample_count": 240.0, "total_gap_seconds": 0.0, "max_gap_seconds": 18.003, "gap_fraction": 0.0,
            "mean": 0.16315092273056508, "std": 0.0002990954011645764, "var": 8.945805899779891e-08,
            "skew": -0.8004145829526675, "kurtosis": -0.11334893487122066, "min": 0.16238415241241455,
            "max": 0.1636175811290741, "mean_abs_change": 1.9203178846656376e-05, "n_peaks": 13.0,
            "smooth_n_peaks": 9.0, "diff_peaks": 26.0, "diff_var": 3.0521698730639922e-09, "diff2_peaks": 51.0,
            "diff2_var": 6.212391530495283e-09, "slope": 4.771337609382082e-06, "mahalanobis": 1.4215588623584612,
            "seconds_since_last_tc": 519.377, "tc_count_in_window": 57.0,
        },
    },
    {
        "channel_id": "channel_23", "label": "nominal", "window_start": "2002-04-25 22:00:00",
        "features": {
            "sample_count": 240.0, "total_gap_seconds": 0.0, "max_gap_seconds": 18.003, "gap_fraction": 0.0,
            "mean": 0.8933611795306206, "std": 6.234574896978494e-05, "var": 3.88699241460344e-09,
            "skew": 0.14655072640534597, "kurtosis": -0.863561071050055, "min": 0.893252432346344,
            "max": 0.893510103225708, "mean_abs_change": 8.947286174524969e-06, "n_peaks": 13.0,
            "smooth_n_peaks": 9.0, "diff_peaks": 26.0, "diff_var": 6.462212908784057e-10, "diff2_peaks": 52.0,
            "diff2_var": 1.2947783548425484e-09, "slope": 8.154014202422666e-07, "mahalanobis": 0.9720468355453991,
            "seconds_since_last_tc": 556.85, "tc_count_in_window": 57.0,
        },
    },
    {
        "channel_id": "channel_24", "label": "nominal", "window_start": "2003-06-25 03:00:00",
        "features": {
            "sample_count": 240.0, "total_gap_seconds": 0.0, "max_gap_seconds": 18.003, "gap_fraction": 0.0,
            "mean": 0.1702415433526039, "std": 8.333752762102028e-05, "var": 6.9451435099843195e-09,
            "skew": 0.10314044362782682, "kurtosis": -0.5595263891661317, "min": 0.17007160186767578,
            "max": 0.17041701078414917, "mean_abs_change": 1.248462715340619e-05, "n_peaks": 11.0,
            "smooth_n_peaks": 7.0, "diff_peaks": 26.0, "diff_var": 1.307828185319043e-09, "diff2_peaks": 49.0,
            "diff2_var": 2.629369382155959e-09, "slope": 8.613576161603297e-07, "mahalanobis": 1.0231162538619878,
            "seconds_since_last_tc": 494.451, "tc_count_in_window": 66.0,
        },
    },
    {
        "channel_id": "channel_24", "label": "nominal", "window_start": "2002-05-06 12:00:00",
        "features": {
            "sample_count": 240.0, "total_gap_seconds": 0.0, "max_gap_seconds": 18.003, "gap_fraction": 0.0,
            "mean": 0.1639253653585911, "std": 7.278924704415325e-05, "var": 5.298274485254772e-09,
            "skew": -0.1718292985931623, "kurtosis": -1.047643010859067, "min": 0.16377373039722443,
            "max": 0.16404499113559723, "mean_abs_change": 1.1553342018894215e-05, "n_peaks": 11.0,
            "smooth_n_peaks": 8.0, "diff_peaks": 27.0, "diff_var": 1.0056853348405598e-09, "diff2_peaks": 50.0,
            "diff2_var": 2.023385815799329e-09, "slope": 5.624397637628525e-09, "mahalanobis": 1.1602427976319925,
            "seconds_since_last_tc": 555.261, "tc_count_in_window": 67.0,
        },
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--delay", type=float, default=3.0, help="seconds between publishes")
    parser.add_argument("--topic", default=TOPIC)
    args = parser.parse_args()

    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, args.topic)

    run_tag = int(time.time())
    for i, window in enumerate(WINDOWS):
        fragment_id = f"frag-demo-{run_tag}-{i:02d}-{window['label']}"
        window_end = window["window_start"][:-8] + f"{int(window['window_start'][-8:-6]) + 1:02d}:00:00"
        payload = {
            "fragment_id": fragment_id,
            "channel_id": window["channel_id"],
            "t_start": window["window_start"].replace(" ", "T") + "Z",
            "t_end": window_end.replace(" ", "T") + "Z",
            "features": window["features"],
            "context": {"source": "scripts/publish_demo_batch.py", "label": window["label"]},
        }
        message_id = publisher.publish(topic_path, json.dumps(payload).encode("utf-8")).result()
        print(f"[{i + 1}/{len(WINDOWS)}] published {fragment_id} "
              f"({window['label']}, {window['channel_id']}) message_id={message_id}")
        if i < len(WINDOWS) - 1:
            time.sleep(args.delay)

    print(f"\ndone: {len(WINDOWS)} windows published to {args.topic}")
    print("check Firestore incidents/ in a few seconds for gate_distance, gate_verdict, severity, routing_destination")


if __name__ == "__main__":
    main()
