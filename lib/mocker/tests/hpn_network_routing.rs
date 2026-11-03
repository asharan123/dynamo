// SPDX-License-Identifier: Apache-2.0

//! Hand-built decode routing cases through the real replay and KV router.
#![cfg(feature = "replay-bench")]

use dynamo_kv_router::config::KvRouterConfig;
use dynamo_mocker::common::protocols::{MockerConfig, WorkerType};
use dynamo_mocker::loadgen::{MooncakeRow, Trace};
use dynamo_mocker::replay::{
    OfflineDisaggReplayConfig, PerRequestRecord, ReplayCaptureOptions, ReplayDeterminism,
    ReplayRouterMode, SlaThresholds,
    simulate_loaded_trace_disagg_with_router_mode_and_capture_options,
};
use serde_json::{Value, json};

const BLOCK_SIZE: usize = 512;

fn role(
    worker_type: WorkerType,
    workers_per_nic: usize,
    startup_time_s: Option<f64>,
) -> MockerConfig {
    let mut config = MockerConfig::from_value(json!({
        "engine": {
            "backend": "vllm",
            "worker_type": worker_type,
            "block_size": BLOCK_SIZE,
            "num_gpu_blocks": 8192,
            "max_num_batched_tokens": 131072,
            "max_num_seqs": 32,
            "kv_transfer_bandwidth": 25.0,
            "kv_transfer_bytes_per_token": 320000,
            "kv_transfer_timing_mode": "full_prompt",
            "kv_transfer_workers_per_nic": workers_per_nic,
        }
    }))
    .unwrap();
    config.startup_time = startup_time_s;
    config
}

/// `(arrival ms, input tokens, output tokens)`, returned in arrival order.
fn replay(
    requests: &[(f64, usize, usize)],
    num_decode_workers: usize,
    decode_workers_per_nic: usize,
    weight: f64,
) -> Vec<PerRequestRecord> {
    replay_with(
        requests,
        num_decode_workers,
        decode_workers_per_nic,
        None,
        KvRouterConfig {
            transfer_load_weight: weight,
            ..Default::default()
        },
    )
}

fn replay_with(
    requests: &[(f64, usize, usize)],
    num_decode_workers: usize,
    decode_workers_per_nic: usize,
    decode_startup_time_s: Option<f64>,
    router_config: KvRouterConfig,
) -> Vec<PerRequestRecord> {
    let mut next_hash = 0u64;
    let rows = requests
        .iter()
        .map(|&(arrival_ms, input, output)| {
            let blocks = input.div_ceil(BLOCK_SIZE) as u64;
            let hash_ids = (next_hash..next_hash + blocks).collect();
            next_hash += blocks;
            MooncakeRow {
                timestamp: Some(arrival_ms),
                input_length: Some(input),
                output_length: Some(output),
                hash_ids: Some(hash_ids),
                ..Default::default()
            }
        })
        .collect();
    let report = simulate_loaded_trace_disagg_with_router_mode_and_capture_options(
        OfflineDisaggReplayConfig {
            prefill_args: role(WorkerType::Prefill, 1, None),
            decode_args: role(
                WorkerType::Decode,
                decode_workers_per_nic,
                decode_startup_time_s,
            ),
            num_prefill_workers: requests.len(),
            num_decode_workers,
        },
        Some(router_config),
        None,
        Trace::from_mooncake_rows(rows, BLOCK_SIZE).unwrap(),
        1.0,
        ReplayRouterMode::KvRouter,
        ReplayCaptureOptions {
            capture_per_request: true,
            determinism: ReplayDeterminism::CanonicalV1,
            ..Default::default()
        },
        None,
        SlaThresholds::default(),
    )
    .unwrap();
    let mut records = report.per_request;
    records.sort_by(|a, b| a.arrival_time_ms.total_cmp(&b.arrival_time_ms));
    records
}

fn decode_worker(record: &PerRequestRecord) -> usize {
    record.decode_worker_idx.expect("request reached decode")
}

/// One worker per machine. A resident long request leaves machine B heavier in
/// blocks with an idle NIC, while machine A is lighter but receiving a transfer
/// when the probe is routed.
const BUSY_NIC_REQUESTS: [(f64, usize, usize); 3] = [
    (0.0, 32_000, 4_000),
    (2_000.0, 16_000, 4_000),
    (2_300.0, 8_000, 8),
];

#[test]
fn zero_weight_follows_block_load_onto_the_busy_nic() {
    let records = replay(&BUSY_NIC_REQUESTS, 2, 1, 0.0);
    let [resident, in_flight, probe] = records.as_slice() else {
        panic!("expected three records");
    };
    assert_ne!(decode_worker(resident), decode_worker(in_flight));
    // The probe is routed while the second request's KV is still arriving.
    assert!(probe.source_held_ms.unwrap() < in_flight.destination_activated_ms.unwrap());
    assert_eq!(decode_worker(probe), decode_worker(in_flight));
}

#[test]
fn positive_weight_avoids_the_busy_nic_despite_more_blocks() {
    let records = replay(&BUSY_NIC_REQUESTS, 2, 1, 3.0);
    let [resident, in_flight, probe] = records.as_slice() else {
        panic!("expected three records");
    };
    assert_ne!(decode_worker(resident), decode_worker(in_flight));
    assert!(probe.source_held_ms.unwrap() < in_flight.destination_activated_ms.unwrap());
    assert_eq!(decode_worker(probe), decode_worker(resident));
}

#[test]
fn positive_weight_avoids_the_sibling_of_a_receiving_worker() {
    // Three residents fill three of four workers, and the fourth takes a transfer.
    // Moving the light resident across positions finds the arrangement where it
    // shares a NIC with the receiving worker, which block load alone then prefers.
    let requests = |light: usize| -> Vec<(f64, usize, usize)> {
        let mut requests: Vec<(f64, usize, usize)> = (0..3)
            .map(|i| {
                let input = if i == light { 8_000 } else { 32_000 };
                (i as f64 * 2_000.0, input, 4_000)
            })
            .collect();
        requests.push((6_000.0, 16_000, 4_000));
        requests.push((6_300.0, 8_000, 8));
        requests
    };
    let same_nic = |records: &[PerRequestRecord]| {
        let (in_flight, probe) = (&records[3], &records[4]);
        assert!(probe.source_held_ms.unwrap() < in_flight.destination_activated_ms.unwrap());
        decode_worker(probe) / 2 == decode_worker(in_flight) / 2
    };
    let light = (0..3)
        .find(|light| same_nic(&replay(&requests(*light), 4, 2, 0.0)))
        .expect("block load alone must choose the sibling in one arrangement");
    assert!(!same_nic(&replay(&requests(light), 4, 2, 5.0)));
}

#[test]
fn positive_weight_changes_nothing_on_an_idle_network() {
    // Each request is routed long after the previous transfer finished.
    let requests = [(0.0, 8_000, 8), (5_000.0, 8_000, 8), (10_000.0, 8_000, 8)];
    let decode_workers = |weight| -> Vec<usize> {
        replay(&requests, 4, 2, weight)
            .iter()
            .map(decode_worker)
            .collect()
    };
    assert_eq!(decode_workers(0.0), decode_workers(3.0));
}

fn read_decisions(path: &std::path::Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn logging_config(log: &std::path::Path, weight: f64) -> KvRouterConfig {
    KvRouterConfig {
        transfer_load_weight: weight,
        transfer_decision_log: Some(log.to_str().unwrap().to_string()),
        ..Default::default()
    }
}

#[test]
fn decision_log_explains_every_decode_decision() {
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("decisions.jsonl");
    let records = replay_with(&BUSY_NIC_REQUESTS, 2, 1, None, logging_config(&log, 3.0));
    let decisions = read_decisions(&log);
    assert_eq!(decisions.len(), records.len());
    for record in &records {
        let decision = decisions
            .iter()
            .find(|decision| decision["request_id"] == record.uuid.as_str())
            .expect("every request has a decision");
        assert_eq!(
            decision["source_rank"].as_u64().unwrap() as usize,
            record.prefill_worker_idx.unwrap()
        );
        assert_eq!(
            decision["selected_rank"].as_u64().unwrap() as usize,
            decode_worker(record)
        );
        assert!(decision["time_ms"].as_f64().unwrap() >= record.source_held_ms.unwrap() - 1e-6);
        assert!(decision["source_nic"].is_u64() && decision["sending"].is_u64());

        let candidates = decision["candidates"].as_array().unwrap();
        assert_eq!(candidates.len(), 2);
        for candidate in candidates {
            for field in [
                "rank",
                "nic",
                "decode_blocks",
                "active_requests",
                "receiving",
                "network_cost",
                "total_cost",
            ] {
                assert!(candidate[field].is_number(), "{field}: {candidate}");
            }
        }
        // The logged costs must explain the choice the real scorer made.
        let lowest = candidates
            .iter()
            .map(|candidate| candidate["total_cost"].as_f64().unwrap())
            .fold(f64::INFINITY, f64::min);
        let selected = candidates
            .iter()
            .find(|candidate| candidate["rank"] == decision["selected_rank"])
            .expect("selected rank is a candidate");
        assert_eq!(selected["total_cost"].as_f64().unwrap(), lowest);
    }
    // The probe routed during the second request's transfer saw it on that NIC.
    let probe = &decisions[2];
    let busy_rank = decode_worker(&records[1]) as u64;
    let busy = probe["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|candidate| candidate["rank"].as_u64() == Some(busy_rank))
        .unwrap();
    assert_eq!(busy["receiving"].as_u64(), Some(1));
}

#[test]
fn decision_log_does_not_change_the_replay() {
    let dir = tempfile::tempdir().unwrap();
    let requests = serde_json::to_string(&replay(&BUSY_NIC_REQUESTS, 2, 1, 3.0)).unwrap();
    let logged = serde_json::to_string(&replay_with(
        &BUSY_NIC_REQUESTS,
        2,
        1,
        None,
        logging_config(&dir.path().join("decisions.jsonl"), 3.0),
    ))
    .unwrap();
    assert_eq!(requests, logged);
}
