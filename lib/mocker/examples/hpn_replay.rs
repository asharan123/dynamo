// SPDX-License-Identifier: Apache-2.0

//! Runs one deterministic offline P/D replay from a JSON experiment file.
//!
//! ```text
//! cargo run -p dynamo-mocker --features replay-bench --example hpn_replay -- <experiment.json>
//! ```
//!
//! Relative paths in the experiment file resolve against its directory. The
//! run writes `requests.jsonl` and `summary.json` to `out_dir`. Both hold only
//! virtual-time values, so identical inputs must produce identical files.
//!
//! A role whose engine `timing_model` uses provider `"ais"` is timed live by the
//! AISimulate perf model (feature `ais-live`), and the run also writes
//! `timing.json` with each such role's resolved model identity and the source
//! commits `run_hpn.sh` exports. Use
//! `examples/hpn_calibration/run_hpn.sh` for that path.

use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use aisimulate_core::replay::{PerRequestAdmissionRecord, ReplayRequestPool};
use anyhow::{Context, Result, bail};
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_mocker::common::protocols::{MockerConfig, WorkerType};
use dynamo_mocker::loadgen::Trace;
use dynamo_mocker::replay::{
    OfflineDisaggReplayConfig, PerRequestRecord, ReplayCaptureOptions, ReplayDeterminism,
    ReplayRouterMode, SlaThresholds, TraceSimulationReport,
    simulate_loaded_trace_disagg_with_router_mode_and_capture_options,
};
use serde::Deserialize;
use serde_json::{Value, json};

#[path = "hpn_replay/ais_timing.rs"]
mod ais_timing;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Experiment {
    trace: PathBuf,
    out_dir: PathBuf,
    block_size: usize,
    num_prefill_workers: usize,
    num_decode_workers: usize,
    /// `MockerConfig` engine fields shared by both roles.
    engine: serde_json::Map<String, Value>,
    /// Role-specific engine fields, applied over `engine`.
    #[serde(default)]
    prefill_engine: serde_json::Map<String, Value>,
    #[serde(default)]
    decode_engine: serde_json::Map<String, Value>,
    /// Partial `KvRouterConfig`; omitted fields keep their defaults.
    #[serde(default)]
    router: Option<Value>,
}

/// Builds one role's engine config. When the role uses live AIS timing, also
/// returns the resolved perf-model identity for `timing.json`.
fn role_config(
    experiment: &Experiment,
    worker_type: WorkerType,
    base: &Path,
) -> Result<(MockerConfig, Option<Value>)> {
    let mut engine = experiment.engine.clone();
    engine.extend(match worker_type {
        WorkerType::Prefill => experiment.prefill_engine.clone(),
        _ => experiment.decode_engine.clone(),
    });
    engine.insert(
        "worker_type".to_string(),
        serde_json::to_value(worker_type)?,
    );
    engine.insert("block_size".to_string(), json!(experiment.block_size));
    ais_timing::inject_worker_type(&mut engine, worker_type)?;
    ais_timing::resolve_systems_paths(&mut engine, base)?;
    let mut args = MockerConfig::from_value(json!({ "engine": engine }))?;
    let identity = ais_timing::install(&mut args)?;
    Ok((args, identity))
}

fn percentile(sorted: &[f64], quantile: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let rank = (quantile * sorted.len() as f64).ceil() as usize;
    Some(sorted[rank.saturating_sub(1).min(sorted.len() - 1)])
}

fn stats(mut values: Vec<f64>) -> Value {
    values.sort_by(f64::total_cmp);
    let mean = (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64);
    json!({
        "count": values.len(),
        "mean": mean,
        "p50": percentile(&values, 0.50),
        "p99": percentile(&values, 0.99),
        "max": values.last(),
    })
}

/// Handoff time from source hold to destination activation. It is
/// `reserve_wait_ms + transfer_ms`.
fn handoff_ms(record: &PerRequestRecord) -> Option<f64> {
    Some(record.destination_activated_ms? - record.source_held_ms?)
}

/// Time a finished prefill waits for a decode destination to reserve memory.
fn reserve_wait_ms(record: &PerRequestRecord) -> Option<f64> {
    Some(record.destination_reserved_ms? - record.source_held_ms?)
}

/// Time the KV transfer spends on the NICs, from reservation to activation.
fn transfer_ms(record: &PerRequestRecord) -> Option<f64> {
    Some(record.destination_activated_ms? - record.destination_reserved_ms?)
}

/// Token-weighted achieved hit rates. `prefill` is the share of prompt tokens
/// the prefill worker reused from its cache. `decode_destination` is the share
/// the decode destination already held, so it did not have to receive them.
/// Destinations count missing tokens in whole blocks, so a prompt counts as its
/// block-rounded length there.
/// Each row is `(input_length, reused_input_tokens, destination_missing_tokens)`.
fn hit_rates(
    rows: impl Iterator<Item = (usize, usize, Option<usize>)>,
    block_size: usize,
) -> Value {
    let (mut input, mut reused) = (0usize, 0usize);
    let (mut handed_off, mut missing) = (0usize, 0usize);
    for (input_length, reused_tokens, missing_tokens) in rows {
        input += input_length;
        reused += reused_tokens;
        if let Some(missing_tokens) = missing_tokens {
            handed_off += input_length.div_ceil(block_size) * block_size;
            missing += missing_tokens;
        }
    }
    let share = |part: usize, whole: usize| (whole > 0).then(|| part as f64 / whole as f64);
    json!({
        "prefill": share(reused, input),
        "decode_destination": share(missing, handed_off).map(|m| 1.0 - m),
    })
}

/// Readmission totals per pool. A readmission is a preempt-and-readmit; the
/// replay report has no direct preemption counter.
fn readmissions<'a>(histories: impl Iterator<Item = &'a [PerRequestAdmissionRecord]>) -> Value {
    let (mut prefill, mut decode, mut requests) = (0usize, 0usize, 0usize);
    for history in histories {
        let mut readmitted = false;
        for admission in history.iter().filter(|a| a.is_readmission) {
            readmitted = true;
            match admission.pool {
                ReplayRequestPool::Prefill => prefill += 1,
                ReplayRequestPool::Decode => decode += 1,
                ReplayRequestPool::Agg => {}
            }
        }
        requests += usize::from(readmitted);
    }
    json!({ "prefill": prefill, "decode": decode, "requests_readmitted": requests })
}

fn write_outputs(
    out_dir: &Path,
    report: &TraceSimulationReport,
    timing: Option<&Value>,
    block_size: usize,
) -> Result<()> {
    fs::create_dir_all(out_dir)
        .with_context(|| format!("failed to create {}", out_dir.display()))?;
    let mut records: Vec<&PerRequestRecord> = report.per_request.iter().collect();
    records.sort_by(|a, b| {
        a.arrival_time_ms
            .total_cmp(&b.arrival_time_ms)
            .then_with(|| a.uuid.cmp(&b.uuid))
    });

    let mut requests = BufWriter::new(fs::File::create(out_dir.join("requests.jsonl"))?);
    for record in &records {
        serde_json::to_writer(&mut requests, *record)?;
        requests.write_all(b"\n")?;
    }
    requests.flush()?;

    let summary = json!({
        "requests": records.len(),
        "ttft_ms": stats(records.iter().filter_map(|r| r.ttft_ms).collect()),
        "itl_ms": stats(records.iter().filter_map(|r| r.itl_ms).collect()),
        "e2e_ms": stats(records.iter().filter_map(|r| r.e2e_latency_ms).collect()),
        "handoff_ms": stats(records.iter().filter_map(|r| handoff_ms(r)).collect()),
        "reserve_wait_ms": stats(records.iter().filter_map(|r| reserve_wait_ms(r)).collect()),
        "transfer_ms": stats(records.iter().filter_map(|r| transfer_ms(r)).collect()),
        "hit_rate": hit_rates(
            records.iter().map(|r| {
                (r.input_length, r.reused_input_tokens, r.destination_missing_tokens)
            }),
            block_size,
        ),
        "readmissions": readmissions(records.iter().map(|r| r.admission_history.as_slice())),
    });
    fs::write(
        out_dir.join("summary.json"),
        serde_json::to_string_pretty(&summary)? + "\n",
    )?;
    if let Some(timing) = timing {
        fs::write(
            out_dir.join("timing.json"),
            serde_json::to_string_pretty(timing)? + "\n",
        )?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let [_, experiment_path] = args.as_slice() else {
        bail!("usage: hpn_replay <experiment.json>");
    };
    let experiment_path = Path::new(experiment_path);
    let base = experiment_path.parent().unwrap_or(Path::new("."));
    let experiment: Experiment = serde_json::from_slice(
        &fs::read(experiment_path)
            .with_context(|| format!("failed to read {}", experiment_path.display()))?,
    )
    .context("invalid experiment file")?;

    let (prefill_args, prefill_timing) = role_config(&experiment, WorkerType::Prefill, base)?;
    let (decode_args, decode_timing) = role_config(&experiment, WorkerType::Decode, base)?;
    let timing = (prefill_timing.is_some() || decode_timing.is_some()).then(|| {
        json!({
            "sources": ais_timing::source_identity(|key| std::env::var(key).ok()),
            "prefill": prefill_timing,
            "decode": decode_timing,
        })
    });
    let config = OfflineDisaggReplayConfig {
        prefill_args,
        decode_args,
        num_prefill_workers: experiment.num_prefill_workers,
        num_decode_workers: experiment.num_decode_workers,
    };
    let router_config: Option<KvRouterConfig> = experiment
        .router
        .clone()
        .map(serde_json::from_value)
        .transpose()
        .context("invalid router config")?;
    let trace = Trace::from_mooncake(&base.join(&experiment.trace), experiment.block_size)?;

    let report = simulate_loaded_trace_disagg_with_router_mode_and_capture_options(
        config,
        router_config,
        None,
        trace,
        1.0,
        ReplayRouterMode::KvRouter,
        ReplayCaptureOptions {
            capture_per_request: true,
            determinism: ReplayDeterminism::CanonicalV1,
            ..Default::default()
        },
        None,
        SlaThresholds::default(),
    )?;
    write_outputs(
        &base.join(&experiment.out_dir),
        &report,
        timing.as_ref(),
        experiment.block_size,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn admission(pool: ReplayRequestPool, is_readmission: bool) -> PerRequestAdmissionRecord {
        PerRequestAdmissionRecord {
            admission_ordinal: 0,
            pool_admission_ordinal: 0,
            pool,
            at_ms: 0.0,
            reused_input_tokens: 0,
            g1_reused_input_tokens: None,
            host_reused_input_tokens: None,
            is_readmission,
        }
    }

    #[test]
    fn readmissions_count_per_pool_and_per_request() {
        use ReplayRequestPool::{Decode, Prefill};
        let clean = vec![admission(Prefill, false), admission(Decode, false)];
        let preempted = vec![
            admission(Prefill, false),
            admission(Decode, false),
            admission(Decode, true),
            admission(Decode, true),
        ];
        let prefill_retry = vec![admission(Prefill, false), admission(Prefill, true)];
        let histories = [clean, preempted, prefill_retry];
        assert_eq!(
            readmissions(histories.iter().map(Vec::as_slice)),
            json!({"prefill": 1, "decode": 2, "requests_readmitted": 2})
        );
    }

    #[test]
    fn no_readmissions_reports_zeros() {
        assert_eq!(
            readmissions(std::iter::empty()),
            json!({"prefill": 0, "decode": 0, "requests_readmitted": 0})
        );
    }

    #[test]
    fn hit_rates_are_token_weighted() {
        // At block size 512 the handed-off prompts span 1024 and 3072 tokens.
        let rows = [
            (1000, 500, Some(1024)),
            (3000, 3000, Some(0)),
            (1000, 0, None),
        ];
        assert_eq!(
            hit_rates(rows.into_iter(), 512),
            json!({"prefill": 0.7, "decode_destination": 0.75})
        );
    }

    #[test]
    fn cold_destination_has_zero_hit_rate_for_a_partial_last_block() {
        let rows = [(1000, 0, Some(1024))];
        assert_eq!(
            hit_rates(rows.into_iter(), 512)["decode_destination"],
            json!(0.0)
        );
    }

    #[test]
    fn hit_rates_without_tokens_are_null() {
        assert_eq!(
            hit_rates(std::iter::empty(), 512),
            json!({"prefill": null, "decode_destination": null})
        );
    }
}
