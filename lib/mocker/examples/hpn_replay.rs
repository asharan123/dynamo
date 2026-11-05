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
//! `timing.json` with each such role's resolved model identity. Use
//! `examples/hpn_calibration/run_hpn.sh` for that path.

use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

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

/// Handoff time from source hold to destination activation. It equals the
/// transfer time when the destination is reserved without waiting.
fn handoff_ms(record: &PerRequestRecord) -> Option<f64> {
    Some(record.destination_activated_ms? - record.source_held_ms?)
}

fn write_outputs(
    out_dir: &Path,
    report: &TraceSimulationReport,
    timing: Option<&Value>,
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
    let timing = (prefill_timing.is_some() || decode_timing.is_some())
        .then(|| json!({ "prefill": prefill_timing, "decode": decode_timing }));
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
    write_outputs(&base.join(&experiment.out_dir), &report, timing.as_ref())
}
