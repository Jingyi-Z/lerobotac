from typing import Any

import numpy as np
import torch

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    TactileToEnvStateProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    OBS_ENV_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_act import ACTConfig, fold_tactile_into_env_state

_FOLDABLE_STAT_NAMES = ("mean", "std", "min", "max", "q01", "q99", "q10", "q90")


def _build_tactile_env_state_stats(config, dataset_stats):
    """Remap per-sensor dataset stats to flat `observation.environment_state` stats.

    The normalizer expects statistics keyed by the *folded* feature name. This
    flattens each per-sensor stat array and concatenates them in the same order
    the fold uses on the tensors. If anything doesn't line up cleanly, the
    function returns the original dataset_stats unchanged (env-state then passes
    through unnormalized rather than crashing).
    """
    if not dataset_stats or not config.tactile_env_state_keys:
        return dataset_stats
    keys = config.tactile_env_state_keys
    if not all(key in dataset_stats for key in keys):
        return dataset_stats

    expected_dim = sum(
        int(np.prod(s)) if len(s) else 1 for s in config.tactile_env_state_shapes
    )
    merged = {}
    for stat_name in _FOLDABLE_STAT_NAMES:
        if not all(stat_name in dataset_stats[key] for key in keys):
            continue
        parts = [np.asarray(dataset_stats[key][stat_name]).reshape(-1) for key in keys]
        concatenated = np.concatenate(parts)
        if concatenated.shape[0] == expected_dim:
            merged[stat_name] = concatenated

    if "mean" not in merged or "std" not in merged:
        return dataset_stats

    new_stats = dict(dataset_stats)
    new_stats[OBS_ENV_STATE] = merged
    return new_stats


def make_act_pre_post_processors(config: ACTConfig, dataset_stats=None):
    """Build ACT's pre- and post-processing pipelines.

    Pre-processing: rename -> add batch dim -> place on device -> (optional)
    fold tactile into env-state -> normalize.
    Post-processing: unnormalize action -> move to CPU.
    """
    # Idempotent — also called from ACTPolicy.__init__ in Step 8.
    fold_tactile_into_env_state(config)

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
    ]

    norm_stats = dataset_stats
    if config.tactile_as_env_state and config.tactile_env_state_keys:
        input_steps.append(
            TactileToEnvStateProcessorStep(
                sensor_keys=list(config.tactile_env_state_keys),
                sensor_shapes=[list(s) for s in config.tactile_env_state_shapes],
                env_state_key=OBS_ENV_STATE,
                drop_sensor_keys=True,
            )
        )
        norm_stats = _build_tactile_env_state_stats(config, dataset_stats)

    input_steps.append(
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=norm_stats,
            device=config.device,
        )
    )

    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )