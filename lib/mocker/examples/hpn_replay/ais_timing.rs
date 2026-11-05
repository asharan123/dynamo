// SPDX-License-Identifier: Apache-2.0

//! Live AISimulate forward-pass timing for `hpn_replay`.
//!
//! A role opts in with the canonical engine field
//! `timing_model: {type: external, provider: "ais", config: <ForwardPassPerfModelConfig>}`.
//! The model is built only through `ForwardPassPerfModel::best_available`.
//! Without the `ais-live` feature such a config is rejected instead of silently
//! replaying on the polynomial model.

use anyhow::{Result, bail};
use dynamo_mocker::common::protocols::{MockerConfig, WorkerType};
use serde_json::{Map, Value};
use std::path::Path;

/// The AIS perf-model config of a role's engine map, if the role uses one.
fn ais_config_mut(engine: &mut Map<String, Value>) -> Result<Option<&mut Map<String, Value>>> {
    let Some(timing) = engine.get_mut("timing_model") else {
        return Ok(None);
    };
    match timing.get("provider").and_then(Value::as_str) {
        Some("ais") => {}
        Some("aic") => bail!("timing_model provider \"aic\" is a retired name; use \"ais\""),
        _ => return Ok(None),
    }
    match timing.get_mut("config").and_then(Value::as_object_mut) {
        Some(config) => Ok(Some(config)),
        None => bail!("AIS timing_model.config must be a ForwardPassPerfModelConfig object"),
    }
}

/// Sets the perf-model `worker_type` to the replay role, before the engine
/// config is parsed. Rejects a config that names a different role.
pub fn inject_worker_type(engine: &mut Map<String, Value>, role: WorkerType) -> Result<()> {
    let Some(config) = ais_config_mut(engine)? else {
        return Ok(());
    };
    let role = serde_json::to_value(role)?;
    match config.get("worker_type") {
        Some(existing) if *existing != role => bail!(
            "AIS timing_model.config.worker_type is {existing} but this engine is the {role} role"
        ),
        _ => {
            config.insert("worker_type".to_string(), role);
            Ok(())
        }
    }
}

/// Resolves relative AIS `systems_paths` against the experiment directory,
/// like every other relative path in the experiment file.
pub fn resolve_systems_paths(engine: &mut Map<String, Value>, base: &Path) -> Result<()> {
    let Some(paths) = ais_config_mut(engine)?.and_then(|config| config.get_mut("systems_paths"))
    else {
        return Ok(());
    };
    let Some(paths) = paths.as_array_mut() else {
        bail!("AIS systems_paths must be a list of paths");
    };
    for path in paths {
        let Some(raw) = path.as_str() else {
            bail!("AIS systems_paths entries must be strings, got {path}");
        };
        if Path::new(raw).is_relative() {
            *path = Value::String(base.join(raw).to_string_lossy().into_owned());
        }
    }
    Ok(())
}

/// Replaces the role's perf model with the live AIS estimator when the role
/// asks for one. Returns the resolved identity to record next to the outputs.
pub fn install(args: &mut MockerConfig) -> Result<Option<Value>> {
    let Some(config) = args.ais_perf_config().cloned() else {
        return Ok(None);
    };
    live::install(args, config).map(Some)
}

#[cfg(not(feature = "ais-live"))]
mod live {
    use super::*;

    pub fn install(_args: &mut MockerConfig, _config: Value) -> Result<Value> {
        bail!(
            "timing_model provider \"ais\" needs the ais-live feature; run through \
             lib/mocker/examples/hpn_calibration/run_hpn.sh (see its README)"
        )
    }
}

#[cfg(feature = "ais-live")]
mod live {
    //! Adapted from `RustAisCallback` in
    //! `lib/bindings/python/rust/llm/ais_callback.rs`, which this example
    //! cannot depend on because it lives in the Python bindings crate.

    use super::*;
    use aisimulate_core::{
        ForwardPassMetrics, ForwardPassPerfModel, ForwardPassPerfModelConfig,
        ForwardPassPerfReadiness, ScheduledRequestMetrics,
    };
    use anyhow::Context;
    use dynamo_mocker::common::perf_model::{AisCallback, PerfModel};
    use serde_json::json;
    use std::sync::Arc;

    pub struct LiveAis {
        model: ForwardPassPerfModel,
        nextn: u32,
    }

    pub fn install(args: &mut MockerConfig, config: Value) -> Result<Value> {
        let config: ForwardPassPerfModelConfig =
            serde_json::from_value(config).context("invalid AIS perf config")?;
        let live = LiveAis::build(config.clone())?;
        let identity = json!({
            "config": config,
            "provenance": live.model.provenance(),
        });
        args.perf_model = Arc::new(PerfModel::from_ais_callback(Arc::new(live)));
        Ok(identity)
    }

    impl LiveAis {
        pub fn build(config: ForwardPassPerfModelConfig) -> Result<Self> {
            let nextn = config
                .speculation
                .as_ref()
                .map_or(config.nextn, |spec| spec.num_speculative_tokens());
            let model = ForwardPassPerfModel::best_available(config).map_err(|err| {
                anyhow::anyhow!(
                    "AIS model construction failed: {err:#}\n(an import error usually \
                     means the run did not go through hpn_calibration/run_hpn.sh)"
                )
            })?;
            if model.diagnostics().readiness != ForwardPassPerfReadiness::Ready {
                bail!("AIS estimator is not ready; offline replay needs a ready model");
            }
            // Same guard as the bindings: speculative decode has no FPM
            // interpolation support yet.
            if nextn > 0
                && model.provenance().is_some_and(|p| {
                    p.selected_estimation_mode == aisimulate_core::EstimationMode::FpmInterpolation
                })
            {
                bail!("AIS FPM interpolation does not support speculative decode; use op_level");
            }
            Ok(Self { model, nextn })
        }

        fn estimate(&self, metrics: ForwardPassMetrics) -> Result<f64> {
            let value = self
                .model
                .estimate_forward_pass_time_ms(&[metrics])?
                .context("AIS estimator returned no estimate")?;
            anyhow::ensure!(
                value.is_finite() && value >= 0.0,
                "AIS returned invalid latency {value}"
            );
            Ok(value)
        }
    }

    impl AisCallback for LiveAis {
        fn predict_prefill(&self, batch_size: usize, new: usize, prefix: usize) -> Result<f64> {
            self.estimate(prefill_metrics(batch_size, new, prefix)?)
        }

        fn predict_decode(&self, batch_size: usize, isl: usize, osl: usize) -> Result<f64> {
            // The mocker asks for one step with osl=2, i.e. context isl+1,
            // matching the bindings' convention.
            let mut total = 0.0;
            let stride = aisimulate_core::perfmodel::engine::DEFAULT_STATIC_STRIDE as usize;
            for step in (1..osl).step_by(stride) {
                let context = isl
                    .checked_add(step)
                    .context("AIS decode context overflow")?;
                total += self.estimate(decode_metrics(batch_size, context, self.nextn)?)?
                    * (osl - step).min(stride) as f64;
            }
            Ok(total)
        }
    }

    fn count(value: usize, field: &str) -> Result<u32> {
        u32::try_from(value).with_context(|| format!("AIS {field} exceeds u32"))
    }

    pub fn prefill_metrics(batch: usize, new: usize, prefix: usize) -> Result<ForwardPassMetrics> {
        let batch = count(batch, "batch size")?;
        let new = count(new, "prefill tokens")?;
        let prefix = count(prefix, "prefix tokens")?;
        Ok(ForwardPassMetrics {
            scheduled_requests: ScheduledRequestMetrics {
                num_prefill_requests: batch,
                sum_prefill_tokens: batch
                    .checked_mul(new)
                    .context("AIS prefill total exceeds u32")?,
                sum_prefill_kv_tokens: batch
                    .checked_mul(prefix)
                    .context("AIS prefix total exceeds u32")?,
                ..Default::default()
            },
            ..Default::default()
        })
    }

    pub fn decode_metrics(batch: usize, context: usize, nextn: u32) -> Result<ForwardPassMetrics> {
        let batch = count(batch, "batch size")?
            .checked_mul(nextn + 1)
            .context("AIS verification batch exceeds u32")?;
        let context = count(context, "decode context")?;
        Ok(ForwardPassMetrics {
            scheduled_requests: ScheduledRequestMetrics {
                num_decode_requests: batch,
                sum_decode_kv_tokens: batch
                    .checked_mul(context)
                    .context("AIS decode KV total exceeds u32")?,
                ..Default::default()
            },
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ais_engine(config: Value) -> Map<String, Value> {
        json!({
            "timing_model": {"type": "external", "provider": "ais", "config": config}
        })
        .as_object()
        .unwrap()
        .clone()
    }

    fn qwen_config() -> Value {
        json!({
            "model": "Qwen/Qwen3-32B", "system": "a100_sxm", "backend": "vllm",
            "backend_version": "current", "tp": 4,
            "estimation_mode": "op_level", "fallback_policy": "deny"
        })
    }

    #[test]
    fn worker_type_comes_from_the_replay_role() {
        let mut engine = ais_engine(qwen_config());
        inject_worker_type(&mut engine, WorkerType::Decode).unwrap();
        assert_eq!(
            engine["timing_model"]["config"]["worker_type"],
            json!("decode")
        );
    }

    #[test]
    fn matching_worker_type_is_accepted() {
        let mut config = qwen_config();
        config["worker_type"] = json!("prefill");
        let mut engine = ais_engine(config);
        inject_worker_type(&mut engine, WorkerType::Prefill).unwrap();
        assert_eq!(
            engine["timing_model"]["config"]["worker_type"],
            json!("prefill")
        );
    }

    #[test]
    fn conflicting_worker_type_is_rejected() {
        let mut config = qwen_config();
        config["worker_type"] = json!("prefill");
        let mut engine = ais_engine(config);
        let err = inject_worker_type(&mut engine, WorkerType::Decode).unwrap_err();
        assert!(err.to_string().contains("worker_type"), "{err}");
    }

    #[test]
    fn non_ais_timing_is_left_alone() {
        let mut engine = json!({"timing_model": {"type": "polynomial"}})
            .as_object()
            .unwrap()
            .clone();
        let before = engine.clone();
        inject_worker_type(&mut engine, WorkerType::Decode).unwrap();
        assert_eq!(engine, before);

        let mut empty = Map::new();
        inject_worker_type(&mut empty, WorkerType::Prefill).unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn legacy_aic_provider_is_rejected() {
        let mut engine = ais_engine(qwen_config());
        engine["timing_model"]["provider"] = json!("aic");
        let err = inject_worker_type(&mut engine, WorkerType::Prefill).unwrap_err();
        assert!(err.to_string().contains("\"ais\""), "{err}");
    }

    #[test]
    fn non_object_ais_config_is_rejected() {
        let mut engine = ais_engine(json!("Qwen/Qwen3-32B"));
        assert!(inject_worker_type(&mut engine, WorkerType::Prefill).is_err());
    }

    #[test]
    fn relative_systems_paths_resolve_against_experiment_dir() {
        let mut config = qwen_config();
        config["systems_paths"] = json!(["../systems", "/abs/systems"]);
        let mut engine = ais_engine(config);
        resolve_systems_paths(&mut engine, Path::new("/exp/dir")).unwrap();
        assert_eq!(
            engine["timing_model"]["config"]["systems_paths"],
            json!(["/exp/dir/../systems", "/abs/systems"])
        );
    }

    #[test]
    fn non_string_systems_path_is_rejected() {
        let mut config = qwen_config();
        config["systems_paths"] = json!([3]);
        let mut engine = ais_engine(config);
        assert!(resolve_systems_paths(&mut engine, Path::new("/exp")).is_err());
    }

    #[test]
    fn polynomial_role_installs_nothing() {
        let mut args = MockerConfig::from_value(json!({"engine": {}})).unwrap();
        assert!(install(&mut args).unwrap().is_none());
    }

    #[cfg(not(feature = "ais-live"))]
    #[test]
    fn ais_role_without_feature_fails_loudly() {
        let mut engine = ais_engine(qwen_config());
        engine.insert("worker_type".into(), json!("prefill"));
        inject_worker_type(&mut engine, WorkerType::Prefill).unwrap();
        let mut args = MockerConfig::from_value(json!({"engine": engine})).unwrap();
        let err = install(&mut args).unwrap_err().to_string();
        assert!(err.contains("ais-live"), "{err}");
        assert!(err.contains("run_hpn.sh"), "{err}");
    }

    #[cfg(feature = "ais-live")]
    mod live_tests {
        use super::super::live::{LiveAis, decode_metrics, prefill_metrics};
        use super::*;
        use aisimulate_core::ForwardPassPerfModelConfig;
        use dynamo_mocker::common::perf_model::AisCallback;

        fn role_config(role: &str) -> ForwardPassPerfModelConfig {
            let mut config = qwen_config();
            config["worker_type"] = json!(role);
            serde_json::from_value(config).unwrap()
        }

        #[test]
        fn metrics_keep_prefix_and_batch() {
            let prefill = prefill_metrics(7, 96, 32).unwrap().scheduled_requests;
            assert_eq!(prefill.num_prefill_requests, 7);
            assert_eq!(prefill.sum_prefill_tokens, 672);
            assert_eq!(prefill.sum_prefill_kv_tokens, 224);
            let decode = decode_metrics(7, 129, 0).unwrap().scheduled_requests;
            assert_eq!(decode.num_decode_requests, 7);
            assert_eq!(decode.sum_decode_kv_tokens, 903);
            assert!(prefill_metrics(u32::MAX as usize, 2, 0).is_err());
        }

        #[test]
        fn cached_tokens_make_prefill_cheaper_than_cold() {
            let model = LiveAis::build(role_config("prefill")).unwrap();
            let cold = model.predict_prefill(1, 32768, 0).unwrap();
            let warm = model.predict_prefill(1, 3277, 29491).unwrap();
            assert!(cold > 3000.0 && cold < 5000.0, "cold 32K prefill {cold} ms");
            assert!(warm < cold / 4.0, "warm {warm} ms vs cold {cold} ms");
        }

        #[test]
        fn unknown_model_is_rejected_with_setup_hint() {
            let mut config = role_config("decode");
            config.model = "not-a-real/model".to_string();
            let err = format!("{:#}", LiveAis::build(config).err().unwrap());
            assert!(err.contains("run_hpn.sh"), "{err}");
        }

        #[test]
        fn install_records_identity() {
            let mut engine = ais_engine(qwen_config());
            engine.insert("worker_type".into(), json!("decode"));
            inject_worker_type(&mut engine, WorkerType::Decode).unwrap();
            let mut args = MockerConfig::from_value(json!({"engine": engine})).unwrap();
            let identity = install(&mut args).unwrap().unwrap();
            assert_eq!(identity["config"]["worker_type"], json!("decode"));
            assert_eq!(identity["config"]["model"], json!("Qwen/Qwen3-32B"));
            assert!(!identity["provenance"].is_null());
        }
    }
}
