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

use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_mocker::common::perf_model::{AisCallback, PerfModel};
use dynamo_mocker::common::protocols::{MockerConfig, WorkerType};
use dynamo_mocker::loadgen::Trace;
use dynamo_mocker::replay::{
    OfflineDisaggReplayConfig, PerRequestRecord, ReplayCaptureOptions, ReplayDeterminism,
    ReplayRouterMode, SlaThresholds, TraceSimulationReport,
    simulate_loaded_trace_disagg_with_router_mode_and_capture_options,
};
use serde::Deserialize;
use serde_json::{Value, json};

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
    /// Experiment-only table with prefill axes (batch, new, cached) and decode
    /// axes (batch, context). Generated from the canonical FPM API.
    #[serde(default)]
    timing_table: Option<PathBuf>,
}

#[derive(Clone, Deserialize)]
struct TimingTable {
    model: String,
    system: String,
    backend: String,
    backend_version: String,
    tp: usize,
    prefill_batch: Vec<usize>,
    prefill_new: Vec<usize>,
    prefill_cached: Vec<usize>,
    prefill_ms: Vec<Vec<Vec<f64>>>,
    decode_batch: Vec<usize>,
    decode_context: Vec<usize>,
    decode_ms: Vec<Vec<f64>>,
}

impl TimingTable {
    fn validate(&self) -> Result<()> {
        fn axis(name: &str, values: &[usize]) -> Result<()> {
            ensure!(
                values.len() >= 2 && values.windows(2).all(|pair| pair[0] < pair[1]),
                "{name} must contain at least two strictly increasing points"
            );
            Ok(())
        }
        for (name, values) in [
            ("prefill_batch", &self.prefill_batch),
            ("prefill_new", &self.prefill_new),
            ("prefill_cached", &self.prefill_cached),
            ("decode_batch", &self.decode_batch),
            ("decode_context", &self.decode_context),
        ] {
            axis(name, values)?;
        }
        ensure!(self.tp > 0, "timing table tp must be positive");
        ensure!(
            self.prefill_ms.len() == self.prefill_batch.len()
                && self.prefill_ms.iter().all(|layer| {
                    layer.len() == self.prefill_new.len()
                        && layer.iter().all(|row| {
                            row.len() == self.prefill_cached.len()
                                && row.iter().all(|ms| ms.is_finite() && *ms >= 0.0)
                        })
                }),
            "prefill timing table has invalid shape or latency"
        );
        ensure!(
            self.decode_ms.len() == self.decode_batch.len()
                && self.decode_ms.iter().all(|row| {
                    row.len() == self.decode_context.len()
                        && row.iter().all(|ms| ms.is_finite() && *ms >= 0.0)
                }),
            "decode timing table has invalid shape or latency"
        );
        Ok(())
    }
}

fn bracket(axis: &[usize], value: usize) -> Result<(usize, usize, f64)> {
    if axis.len() < 2 || value < axis[0] || value > *axis.last().unwrap() {
        bail!("timing query {value} is outside table axis {axis:?}");
    }
    match axis.binary_search(&value) {
        Ok(index) => Ok((index, index, 0.0)),
        Err(upper) => {
            let lower = upper - 1;
            let fraction = (value - axis[lower]) as f64 / (axis[upper] - axis[lower]) as f64;
            Ok((lower, upper, fraction))
        }
    }
}

fn lerp(a: f64, b: f64, fraction: f64) -> f64 {
    a + (b - a) * fraction
}

impl AisCallback for TimingTable {
    fn predict_prefill(&self, batch: usize, new: usize, cached: usize) -> Result<f64> {
        let (b0, b1, bf) = bracket(&self.prefill_batch, batch)?;
        let (n0, n1, nf) = bracket(&self.prefill_new, new)?;
        let (c0, c1, cf) = bracket(&self.prefill_cached, cached)?;
        let at =
            |b: usize, n: usize| lerp(self.prefill_ms[b][n][c0], self.prefill_ms[b][n][c1], cf);
        Ok(lerp(
            lerp(at(b0, n0), at(b0, n1), nf),
            lerp(at(b1, n0), at(b1, n1), nf),
            bf,
        ))
    }

    fn predict_decode(&self, batch: usize, context: usize, _osl: usize) -> Result<f64> {
        let (b0, b1, bf) = bracket(&self.decode_batch, batch)?;
        let (c0, c1, cf) = bracket(&self.decode_context, context)?;
        Ok(lerp(
            lerp(self.decode_ms[b0][c0], self.decode_ms[b0][c1], cf),
            lerp(self.decode_ms[b1][c0], self.decode_ms[b1][c1], cf),
            bf,
        ))
    }
}

fn role_config(
    experiment: &Experiment,
    worker_type: WorkerType,
    base: &Path,
) -> Result<MockerConfig> {
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
    let table = experiment
        .timing_table
        .as_ref()
        .map(|path| -> Result<TimingTable> {
            let table: TimingTable = serde_json::from_slice(&fs::read(base.join(path))?)?;
            table.validate()?;
            Ok(table)
        })
        .transpose()?;
    if let Some(table) = &table {
        engine.insert(
            "timing_model".to_string(),
            json!({
                "type": "external", "provider": "ais", "config": {
                    "model": table.model, "system": table.system,
                    "backend": table.backend, "backend_version": table.backend_version,
                    "worker_type": worker_type, "tp": table.tp
                }
            }),
        );
    }
    let mut config = MockerConfig::from_value(json!({ "engine": engine }))?;
    if let Some(table) = table {
        config.perf_model = Arc::new(PerfModel::from_ais_callback(Arc::new(table)));
    }
    Ok(config)
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

fn write_outputs(out_dir: &Path, report: &TraceSimulationReport) -> Result<()> {
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

    let config = OfflineDisaggReplayConfig {
        prefill_args: role_config(&experiment, WorkerType::Prefill, base)?,
        decode_args: role_config(&experiment, WorkerType::Decode, base)?,
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
    write_outputs(&base.join(&experiment.out_dir), &report)
}

#[cfg(test)]
mod tests {
    use super::TimingTable;
    use dynamo_mocker::common::perf_model::AisCallback;

    fn table() -> TimingTable {
        TimingTable {
            model: "test".into(),
            system: "test".into(),
            backend: "vllm".into(),
            backend_version: "current".into(),
            tp: 1,
            prefill_batch: vec![1, 3],
            prefill_new: vec![10, 30],
            prefill_cached: vec![0, 20],
            prefill_ms: vec![
                vec![vec![10.0, 30.0], vec![30.0, 50.0]],
                vec![vec![30.0, 50.0], vec![50.0, 70.0]],
            ],
            decode_batch: vec![1, 3],
            decode_context: vec![10, 30],
            decode_ms: vec![vec![10.0, 30.0], vec![30.0, 50.0]],
        }
    }

    #[test]
    fn timing_table_interpolates_all_prefill_inputs() {
        let table = table();
        table.validate().unwrap();
        assert_eq!(table.predict_prefill(2, 20, 10).unwrap(), 40.0);
        assert_eq!(table.predict_decode(2, 20, 2).unwrap(), 30.0);
        assert!(table.predict_prefill(2, 31, 10).is_err());
    }

    #[test]
    fn timing_table_rejects_malformed_grid() {
        let mut table = table();
        table.prefill_ms[0][0].pop();
        assert!(table.validate().is_err());
    }
}
