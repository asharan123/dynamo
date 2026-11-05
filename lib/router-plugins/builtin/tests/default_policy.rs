// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod support;
use dynamo_custom_policy_builtin::{DefaultWorkerSelector, default_policy, default_registry};
use dynamo_kv_router::protocols::WorkerWithDpRank;
use dynamo_kv_router::scheduling::TransferLoad;
use dynamo_kv_router::{
    KvRouterConfig, RoutingPartitionRef, WorkerInputView, WorkerInputs, WorkerPicker,
    WorkerSelectionContext, WorkerSelectionPolicy, WorkerSelectionPolicyError, WorkerSelector,
    WorkerType,
};
use support::*;

#[test]
fn seeded_selection_matches_reference_across_cache_and_load_shapes() {
    for temperature in [0.0, 0.7] {
        for prompt in [1, 17, 127, 2048] {
            for mode in 0..32 {
                let (workers, mut request) = fixture(16, prompt);
                let config = KvRouterConfig {
                    router_temperature: temperature,
                    overlap_score_credit_decay: 0.6,
                    host_cache_hit_weight: 0.25,
                    disk_cache_hit_weight: 0.1,
                    decode_active_request_weight: if mode & 8 == 0 { 0.0 } else { 0.7 },
                    shared_cache_multiplier: Some(if mode & 16 == 0 { 0.0 } else { 0.6 }),
                    ..Default::default()
                };
                if mode % 8 >= 4 {
                    request.shared_cache_hits =
                        Some(dynamo_kv_router::SharedCacheHits::from_ranges(vec![
                            1..3,
                            5..12,
                        ]));
                }
                match mode % 4 {
                    1 => request.overlap.tier_overlap_blocks = Default::default(),
                    2 => request.worker_loads.clear(),
                    3 => request.track_prefill_tokens = false,
                    _ => {}
                }
                let reference = dynamo_kv_router::DefaultWorkerSelector::new_seeded(
                    Some(config.clone()),
                    "test",
                    42,
                );
                let plugin = DefaultWorkerSelector::new_seeded(Some(config), "test", 42);
                for _ in 0..64 {
                    let input = support::selection_input(&workers, &request, 16);
                    let expected = reference.select_worker(input).unwrap();
                    let actual = plugin.select_worker(input).unwrap();
                    assert_eq!(
                        actual.worker, expected.worker,
                        "temperature={temperature} prompt={prompt} mode={mode}"
                    );
                    assert_eq!(actual.cached_tokens, expected.cached_tokens);
                    assert_eq!(
                        actual.potential_decode_blocks,
                        expected.potential_decode_blocks
                    );
                }
            }
        }
    }
}

#[test]
fn empty_prompt_routes_by_load() {
    // Workers 0 and 1 sit on the prefill floor with decode backlog. Worker 2 is one
    // prefill block above the floor with no decode backlog, so it must win.
    let (workers, mut request) = fixture(3, 0);
    request.overlap = Default::default();
    for (worker, load) in &mut request.worker_loads {
        let unloaded = worker.worker_id == 2;
        *load = dynamo_kv_router::WorkerLoadProjection {
            active_prefill_tokens: if unloaded { 1616 } else { 1600 },
            active_decode_blocks: if unloaded { 0 } else { 50 },
            ..Default::default()
        };
    }
    for overlap_score_credit_decay in [0.0, 1.0] {
        let config = KvRouterConfig {
            router_temperature: 0.0,
            overlap_score_credit_decay,
            ..Default::default()
        };
        let plugin = DefaultWorkerSelector::new_seeded(Some(config), "test", 42);
        let result = plugin
            .select_worker(support::selection_input(&workers, &request, 16))
            .unwrap();
        assert_eq!(
            result.worker.worker_id, 2,
            "decay={overlap_score_credit_decay}"
        );
    }
}

#[test]
fn unseeded_sampling_matches_reference_with_the_same_random_draw() {
    for count in [1, 8, 64] {
        for temperature in [0.1, 0.7, 1.0, 2.0] {
            for equal_costs in [false, true] {
                let (workers, mut request) = fixture(count, 2048);
                if equal_costs {
                    request.worker_loads.clear();
                    request.overlap = Default::default();
                }
                let config = KvRouterConfig {
                    router_temperature: temperature,
                    ..Default::default()
                };
                let reference =
                    dynamo_kv_router::DefaultWorkerSelector::new(Some(config.clone()), "prefill");
                let plugin = default_policy(config, "prefill");
                for seed in 0..64 {
                    let input = selection_input(&workers, &request, 16);
                    fastrand::seed(seed);
                    let expected = reference.select_worker(input).unwrap();
                    let next_random = fastrand::u64(..);
                    fastrand::seed(seed);
                    let actual = plugin.select_worker(input).unwrap();
                    assert_eq!(actual.worker, expected.worker);
                    assert_eq!(actual.cached_tokens, expected.cached_tokens);
                    assert_eq!(fastrand::u64(..), next_random);
                }
            }
        }
    }
}

#[test]
fn unseeded_minimum_picker_only_selects_workers_tied_for_lowest_cost() {
    let (workers, mut request) = fixture(8, 17);
    let policy = default_policy(
        KvRouterConfig {
            overlap_score_credit: 0.0,
            prefill_load_scale: 0.0,
            decode_active_request_weight: 1.0,
            ..Default::default()
        },
        "prefill",
    );
    for best in [[0, 1], [2, 5], [6, 7]] {
        for (worker, load) in &mut request.worker_loads {
            load.active_requests = if best.contains(&worker.worker_id) {
                0
            } else {
                10
            };
            load.active_decode_blocks = 0;
            load.additional_active_blocks = 0;
        }
        for _ in 0..32 {
            let selected = policy
                .select_worker(support::selection_input(&workers, &request, 16))
                .unwrap();
            assert!(best.contains(&selected.worker.worker_id));
        }
    }
}

#[test]
fn prepared_values_follow_each_request_when_reusing_a_policy() {
    for label in ["prefill", "decode"] {
        for temperature in [0.0, 0.7] {
            let config = KvRouterConfig {
                router_temperature: temperature,
                overlap_score_credit_decay: 0.6,
                host_cache_hit_weight: 0.25,
                disk_cache_hit_weight: 0.1,
                ..Default::default()
            };
            let reference = dynamo_kv_router::DefaultWorkerSelector::new_seeded(
                Some(config.clone()),
                label,
                42,
            );
            let plugin = DefaultWorkerSelector::new_seeded(Some(config), label, 42);
            for round in 0..64 {
                let prompt = [1, 17, 127, 2048][round % 4];
                let block_size = [8, 16, 32, 17, 3][round % 5];
                let (workers, mut request) = fixture(16, prompt);
                request.track_prefill_tokens = round % 2 == 0;
                if round % 3 == 0 {
                    request.overlap.tier_overlap_blocks = Default::default();
                }
                if round % 5 == 0 {
                    request.worker_loads.clear();
                } else {
                    for load in request.worker_loads.values_mut() {
                        load.active_prefill_tokens += round * 19;
                    }
                }
                let input = support::selection_input(&workers, &request, block_size);
                let expected = reference.select_worker(input).unwrap();
                let actual = plugin.select_worker(input).unwrap();
                assert_eq!(
                    actual.worker, expected.worker,
                    "label={label} temperature={temperature} round={round}"
                );
                assert_eq!(actual.cached_tokens, expected.cached_tokens);
                assert_eq!(
                    actual.potential_decode_blocks,
                    expected.potential_decode_blocks
                );
            }
        }
    }
}

#[test]
fn exact_prompt_and_accounting_inputs_are_available_to_external_pickers() {
    struct Inspect;
    impl WorkerPicker for Inspect {
        fn required_worker_inputs(&self) -> WorkerInputs {
            WorkerInputs::CACHE | WorkerInputs::LOAD
        }
        fn pick(
            &mut self,
            context: &WorkerSelectionContext<'_>,
            input: WorkerInputView<'_>,
        ) -> Result<usize, WorkerSelectionPolicyError> {
            assert_eq!(context.prompt_tokens(), 17);
            assert_eq!(context.request_blocks(), 2);
            assert!(input.load().unwrap().iter().all(|load| load.is_available()));
            for cache in input.cache().unwrap().iter() {
                assert!(cache.has_tier_matches());
                let (blocks, tokens) = cache.accounting_cache_estimate();
                assert_eq!((blocks * 16.0) as usize, tokens);
            }
            Ok(0)
        }
    }
    let (workers, mut request) = fixture(2, 17);
    // Tier availability belongs to the lookup, even when its only entry is ineligible.
    request.overlap.tier_overlap_blocks = Default::default();
    request
        .overlap
        .tier_overlap_blocks
        .disk
        .insert(WorkerWithDpRank::from_worker_id(999), 1);
    WorkerSelectionPolicy::new(KvRouterConfig::default(), "test", vec![], Box::new(Inspect))
        .select_worker(support::selection_input(&workers, &request, 16))
        .unwrap();
}

#[test]
fn configured_default_resolves_for_every_role_and_uses_exclusive_affinity() {
    let registry = default_registry();
    for conditional_disagg_enabled in [false, true] {
        for role in [
            WorkerType::Aggregated,
            WorkerType::Prefill,
            WorkerType::Decode,
            WorkerType::Encode,
        ] {
            let config = KvRouterConfig {
                conditional_disagg_enabled,
                ..Default::default()
            };
            let factory = registry
                .resolve_for_worker_type(&config, role)
                .unwrap()
                .unwrap();
            let policy = factory(&config, role, RoutingPartitionRef::new("model", "default"));
            let inputs =
                <WorkerSelectionPolicy as WorkerSelector<TestWorker>>::required_worker_inputs(
                    &policy,
                );
            assert_eq!(
                inputs.contains(WorkerInputs::CACHE),
                role != WorkerType::Decode || conditional_disagg_enabled
            );
            assert!(
                <WorkerSelectionPolicy as WorkerSelector<TestWorker>>::uses_exclusive_affinity_target(
                    &policy
                )
            );
        }
    }
}

#[test]
fn mandatory_pin_is_preserved() {
    let (workers, mut request) = fixture(4, 17);
    request.pinned_worker = Some(WorkerWithDpRank::new(3, 1));
    let selected = default_policy(KvRouterConfig::default(), "test")
        .select_worker(support::selection_input(&workers, &request, 16))
        .unwrap();
    assert_eq!(Some(selected.worker), request.pinned_worker);
}

#[test]
fn configured_parameters_replace_request_score_overrides() {
    use dynamo_kv_router::RouterConfigOverride;
    let (workers, mut request) = fixture(2, 160);
    for (worker, load) in &mut request.worker_loads {
        load.active_prefill_tokens = 0;
        load.active_requests = 0;
        load.active_decode_blocks = if worker.worker_id == 0 { 8 } else { 0 };
        load.additional_active_blocks = 0;
        request
            .overlap
            .tier_overlap_blocks
            .device
            .insert(*worker, if worker.worker_id == 0 { 10 } else { 0 });
    }
    request.router_config_override = Some(RouterConfigOverride {
        overlap_score_credit: Some(0.0),
        router_temperature: Some(10.0),
        ..Default::default()
    });
    let config = KvRouterConfig {
        overlap_score_credit: 2.0,
        host_cache_hit_weight: 0.0,
        disk_cache_hit_weight: 0.0,
        router_temperature: 0.0,
        ..Default::default()
    };
    let policy = default_policy(config, "prefill");
    for _ in 0..32 {
        let result = policy
            .select_worker(support::selection_input(&workers, &request, 16))
            .unwrap();
        assert_eq!(result.worker.worker_id, 0);
    }
}

#[test]
fn named_default_parameters_resolve_and_reject_invalid_values() {
    use std::io::Write;
    for (parameter, succeeds) in [
        ("overlap_score_credit: 2.0", true),
        ("overlap_score_credit: -1.0", false),
        ("unknown: 1", false),
    ] {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        write!(file, "worker_selection:\n  aggregated: tuned\n  instances:\n    - name: tuned\n      type: dynamo-default-cost-fn\n      parameters:\n        {parameter}\n").unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(file.path().display().to_string()),
            ..Default::default()
        };
        let mut registry = default_registry();
        dynamo_custom_policy_builtin::register(&mut registry).unwrap();
        assert_eq!(registry.resolve(&config).is_ok(), succeeds);
    }
}

#[test]
fn pin_does_not_advance_seeded_random_stream() {
    let (workers, mut request) = fixture(8, 127);
    let config = KvRouterConfig {
        router_temperature: 0.7,
        ..Default::default()
    };
    let reference =
        dynamo_kv_router::DefaultWorkerSelector::new_seeded(Some(config.clone()), "prefill", 42);
    let plugin = DefaultWorkerSelector::new_seeded(Some(config), "prefill", 42);
    for pinned in [true, false, true, false, false] {
        request.pinned_worker = pinned.then_some(WorkerWithDpRank::new(3, 1));
        let input = support::selection_input(&workers, &request, 16);
        assert_eq!(
            reference.select_worker(input).unwrap().worker,
            plugin.select_worker(input).unwrap().worker
        );
    }
}

#[test]
fn yaml_parameters_override_legacy_config_and_omissions_inherit_it() {
    let (workers, mut request) = fixture(2, 160);
    for (worker, load) in &mut request.worker_loads {
        load.active_prefill_tokens = 0;
        load.active_requests = 0;
        load.active_decode_blocks = if worker.worker_id == 0 { 8 } else { 0 };
        load.additional_active_blocks = 0;
        request
            .overlap
            .tier_overlap_blocks
            .device
            .insert(*worker, if worker.worker_id == 0 { 10 } else { 0 });
    }
    let mut registry = default_registry();
    dynamo_custom_policy_builtin::register(&mut registry).unwrap();
    for (parameters, expected_worker, expects_cache) in [
        ("{}", 0, true),
        (
            "{overlap_score_credit: 0, shared_cache_multiplier: 0}",
            1,
            false,
        ),
    ] {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), format!(
            "worker_selection:\n  aggregated: tuned\n  instances:\n    - name: tuned\n      type: dynamo-default-cost-fn\n      parameters: {parameters}\n"
        )).unwrap();
        let config = KvRouterConfig {
            router_policy_config: Some(file.path().display().to_string()),
            overlap_score_credit: 2.0,
            host_cache_hit_weight: 0.0,
            disk_cache_hit_weight: 0.0,
            router_temperature: 0.0,
            ..Default::default()
        };
        let factory = registry.resolve(&config).unwrap().unwrap();
        let policy = factory(
            &config,
            WorkerType::Aggregated,
            RoutingPartitionRef::new("model", "default"),
        );
        assert_eq!(
            <WorkerSelectionPolicy as WorkerSelector<TestWorker>>::required_worker_inputs(&policy)
                .contains(WorkerInputs::CACHE),
            expects_cache,
        );
        let selected = policy
            .select_worker(support::selection_input(&workers, &request, 16))
            .unwrap();
        assert_eq!(selected.worker.worker_id, expected_worker);
    }
}

#[test]
fn shared_cache_credit_defaults_and_legacy_precedence_apply_to_selection() {
    let (workers, mut request) = fixture(2, 64);
    request.overlap = Default::default();
    for (worker, load) in &mut request.worker_loads {
        // With no shared credit, the warm worker costs 3 blocks versus 4.
        // At the default 0.5 credit, it costs 2.5 versus the cold worker's 2.
        *load = Default::default();
        load.active_decode_blocks = if worker.worker_id == 0 { 2 } else { 0 };
        request
            .overlap
            .tier_overlap_blocks
            .device
            .insert(*worker, if worker.worker_id == 0 { 3 } else { 0 });
    }
    #[allow(clippy::single_range_in_vec_init)]
    let hits = dynamo_kv_router::SharedCacheHits::from_ranges(vec![0..4]);
    request.shared_cache_hits = Some(hits);

    let mut registry = default_registry();
    dynamo_custom_policy_builtin::register(&mut registry).unwrap();
    for (cache_type, legacy, parameters, expected_worker) in [
        ("none", None, "{}", 0),
        ("hicache", None, "{}", 1),
        ("hicache", Some(0.0), "{}", 0),
        ("hicache", Some(0.3), "{}", 0),
        ("hicache", Some(0.0), "{shared_cache_multiplier: 0.5}", 1),
        ("hicache", Some(0.5), "{shared_cache_multiplier: 0}", 0),
    ] {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), format!(
            "router:\n  shared_cache_type: {cache_type}\nworker_selection:\n  aggregated: tuned\n  instances:\n    - name: tuned\n      type: dynamo-default-cost-fn\n      parameters: {parameters}\n"
        )).unwrap();
        let mut config = KvRouterConfig {
            router_policy_config: Some(file.path().display().to_string()),
            shared_cache_multiplier: legacy,
            ..Default::default()
        };
        config.apply_policy_config().unwrap();
        let factory = registry.resolve(&config).unwrap().unwrap();
        let policy = factory(
            &config,
            WorkerType::Aggregated,
            RoutingPartitionRef::new("model", "default"),
        );
        let select = |policy: WorkerSelectionPolicy| {
            policy
                .select_worker(support::selection_input(&workers, &request, 16))
                .unwrap()
                .worker
                .worker_id
        };
        assert_eq!(
            select(policy),
            expected_worker,
            "cache={cache_type} legacy={legacy:?} parameters={parameters}"
        );
        if parameters == "{}" {
            assert_eq!(select(default_policy(config, "prefill")), expected_worker);
        }
    }
}

/// A 1600-token prompt at block size 16 is 100 blocks. Rank 0 of workers 0 and 1
/// carry the given block loads; every other rank is priced out of contention.
fn transfer_scoring_selection(
    weight: f64,
    blocks: [usize; 2],
    transfer_load: Option<(usize, [usize; 2])>,
) -> u64 {
    cached_transfer_scoring_selection(weight, blocks, transfer_load, [0, 0])
}

fn cached_transfer_scoring_selection(
    weight: f64,
    blocks: [usize; 2],
    transfer_load: Option<(usize, [usize; 2])>,
    cached_blocks: [usize; 2],
) -> u64 {
    let (workers, mut request) = fixture(2, 1600);
    request.overlap = Default::default();
    request.track_prefill_tokens = false;
    for (worker, load) in &mut request.worker_loads {
        load.active_requests = 0;
        load.active_prefill_tokens = 0;
        load.additional_active_blocks = 0;
        load.active_decode_blocks = if worker.dp_rank == 0 {
            blocks[worker.worker_id as usize]
        } else {
            1_000_000
        };
    }
    request.transfer_load = transfer_load.map(|(sending, receiving)| TransferLoad {
        sending,
        receiving: (0..2u64)
            .map(|id| (WorkerWithDpRank::new(id, 0), receiving[id as usize]))
            .collect(),
        cached_blocks: (0..2u64)
            .map(|id| (WorkerWithDpRank::new(id, 0), cached_blocks[id as usize]))
            .collect(),
    });
    let policy = default_policy(
        KvRouterConfig {
            overlap_score_credit: 0.0,
            transfer_load_weight: weight,
            ..Default::default()
        },
        "replay",
    );
    let selected = policy
        .select_worker(support::selection_input(&workers, &request, 16))
        .unwrap();
    assert_eq!(selected.worker.dp_rank, 0);
    selected.worker.worker_id
}

#[test]
fn transfer_cost_is_weight_times_blocks_times_receive_load() {
    // Worker 0 pays 1 * 100 * 3 = 300 for its busy NIC, so worker 1 wins with
    // 299 blocks of its own and loses with 301.
    assert_eq!(
        transfer_scoring_selection(1.0, [0, 299], Some((0, [3, 0]))),
        1
    );
    assert_eq!(
        transfer_scoring_selection(1.0, [0, 301], Some((0, [3, 0]))),
        0
    );
    // A send load of 1 charges worker 1 for 100 as well, leaving a margin of 200.
    assert_eq!(
        transfer_scoring_selection(1.0, [0, 199], Some((1, [3, 0]))),
        1
    );
    assert_eq!(
        transfer_scoring_selection(1.0, [0, 201], Some((1, [3, 0]))),
        0
    );
    // The weight scales the same cost: 0.5 * 100 * 3 = 150.
    assert_eq!(
        transfer_scoring_selection(0.5, [0, 149], Some((0, [3, 0]))),
        1
    );
    assert_eq!(
        transfer_scoring_selection(0.5, [0, 151], Some((0, [3, 0]))),
        0
    );
}

#[test]
fn busy_send_nic_charges_every_candidate_the_same() {
    // At sending 4, both pay 400 whatever their receive loads below 4, so block
    // load alone decides.
    assert_eq!(
        transfer_scoring_selection(1.0, [5, 6], Some((4, [0, 2]))),
        0
    );
    assert_eq!(
        transfer_scoring_selection(1.0, [6, 5], Some((4, [0, 2]))),
        1
    );
}

#[test]
fn idle_network_and_shared_nic_leave_block_load_to_decide() {
    assert_eq!(
        transfer_scoring_selection(1.0, [6, 5], Some((0, [0, 0]))),
        1
    );
    // Two candidates on one NIC see one receive load.
    assert_eq!(
        transfer_scoring_selection(1.0, [6, 5], Some((0, [3, 3]))),
        1
    );
}

#[test]
fn transfer_cost_counts_only_blocks_missing_at_the_candidate() {
    // Worker 0 holds 90 of 100 blocks, so it pays 1 * 10 * 3 = 30.
    assert_eq!(
        cached_transfer_scoring_selection(1.0, [0, 29], Some((0, [3, 0])), [90, 0]),
        1
    );
    assert_eq!(
        cached_transfer_scoring_selection(1.0, [0, 31], Some((0, [3, 0])), [90, 0]),
        0
    );
}

#[test]
fn full_hit_pays_no_transfer_cost() {
    assert_eq!(
        cached_transfer_scoring_selection(1.0, [50, 49], Some((0, [50, 0])), [100, 0]),
        1
    );
    assert_eq!(
        cached_transfer_scoring_selection(1.0, [49, 50], Some((0, [50, 0])), [100, 0]),
        0
    );
    // An estimate past the prompt still costs nothing rather than a credit.
    assert_eq!(
        cached_transfer_scoring_selection(1.0, [50, 49], Some((0, [50, 0])), [200, 0]),
        1
    );
}

#[test]
fn busy_send_nic_charges_candidates_by_their_missing_blocks() {
    // At sending 4, worker 0 pays 40 and worker 1 pays 400, a margin of 360.
    assert_eq!(
        cached_transfer_scoring_selection(1.0, [359, 0], Some((4, [0, 0])), [90, 0]),
        0
    );
    assert_eq!(
        cached_transfer_scoring_selection(1.0, [361, 0], Some((4, [0, 0])), [90, 0]),
        1
    );
}

#[test]
fn idle_network_ignores_cached_blocks() {
    assert_eq!(
        cached_transfer_scoring_selection(1.0, [6, 5], Some((0, [0, 0])), [100, 0]),
        1
    );
}

#[test]
fn transfer_cost_needs_both_a_weight_and_nic_load() {
    // Without the weight, the lighter worker wins despite its busy NIC.
    assert_eq!(
        transfer_scoring_selection(0.0, [0, 1], Some((0, [50, 0]))),
        0
    );
    // A selection without NIC load, such as prefill, ignores the weight.
    assert_eq!(transfer_scoring_selection(1.0, [0, 1], None), 0);
}
