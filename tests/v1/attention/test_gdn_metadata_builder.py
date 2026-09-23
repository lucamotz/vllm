# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GDNAttentionMetadataBuilder.build() — specifically the
reclassification of non-spec decodes as prefills when spec decodes exist.
Covers the fix for https://github.com/vllm-project/vllm/issues/34845.
"""

from dataclasses import dataclass, fields
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig, VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
    gdn_precompute_metadata,
)
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")


@dataclass
class GDNBuildTestCase:
    """Specification for a GDN metadata builder classification test."""

    seq_lens: list[int]
    query_lens: list[int]
    num_decode_draft_tokens: list[int] | None  # None = no spec config
    num_speculative_tokens: int
    expected_num_decodes: int
    expected_num_prefills: int
    expected_num_prefill_tokens: int
    expected_num_spec_decodes: int


GDN_BUILD_TEST_CASES = {
    # The original #34845 crash: non-spec query_len=1 + spec decode
    "mixed_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[65, 20],
        query_lens=[1, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
    # All requests are spec decodes — no reclassification needed
    "pure_spec_decode": GDNBuildTestCase(
        seq_lens=[50, 30],
        query_lens=[3, 3],
        num_decode_draft_tokens=[2, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=2,
    ),
    # No speculative config at all — standard decode path
    "pure_regular_decode": GDNBuildTestCase(
        seq_lens=[40, 30, 20],
        query_lens=[1, 1, 1],
        num_decode_draft_tokens=None,
        num_speculative_tokens=0,
        expected_num_decodes=3,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=0,
    ),
    # Multi-token prefill alongside spec decode — no decode to reclassify
    "spec_decode_with_real_prefill": GDNBuildTestCase(
        seq_lens=[100, 20],
        query_lens=[50, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=50,
        expected_num_spec_decodes=1,
    ),
    # All three types in one batch — decode gets reclassified
    "prefill_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[100, 65, 20],
        query_lens=[50, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=2,
        expected_num_prefill_tokens=51,
        expected_num_spec_decodes=1,
    ),
    # Multiple non-spec query_len=1 requests all reclassified
    "multiple_decodes_reclassified": GDNBuildTestCase(
        seq_lens=[40, 50, 60, 20],
        query_lens=[1, 1, 1, 3],
        num_decode_draft_tokens=[-1, -1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=3,
        expected_num_prefill_tokens=3,
        expected_num_spec_decodes=1,
    ),
    # Zero-length padded sequence excluded from counts
    "zero_length_padding_with_spec": GDNBuildTestCase(
        seq_lens=[16, 65, 20],
        query_lens=[0, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
}


def _create_gdn_builder(
    num_speculative_tokens: int = 0,
    full_cuda_graph: bool = False,
) -> GDNAttentionMetadataBuilder:
    """Create a GDNAttentionMetadataBuilder with minimal config."""
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
    )
    if full_cuda_graph:
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    if num_speculative_tokens > 0:
        vllm_config.speculative_config = SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=num_speculative_tokens,
        )
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
    )
    return GDNAttentionMetadataBuilder(
        kv_cache_spec=mamba_spec,
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )


def _build(
    builder: GDNAttentionMetadataBuilder,
    batch_spec: BatchSpec,
    num_decode_draft_tokens: list[int] | None = None,
) -> GDNAttentionMetadata:
    """Build GDN attention metadata, optionally with spec-decode kwargs."""
    common = create_common_attn_metadata(batch_spec, BLOCK_SIZE, DEVICE)
    kwargs: dict = {}
    if num_decode_draft_tokens is not None:
        num_decode_draft_tokens_cpu = torch.tensor(
            num_decode_draft_tokens, dtype=torch.int32
        )
        num_accepted_tokens = torch.ones(
            batch_spec.batch_size, dtype=torch.int32, device=DEVICE
        )
        num_accepted_tokens, gdn_precomputed_metadata = gdn_precompute_metadata(
            num_decode_draft_tokens_cpu,
            num_accepted_tokens,
            common.query_start_loc,
            common.query_start_loc_cpu,
            builder.num_spec,
        )
        kwargs["num_decode_draft_tokens_cpu"] = num_decode_draft_tokens_cpu
        kwargs["num_accepted_tokens"] = num_accepted_tokens
        kwargs["gdn_precomputed_metadata"] = gdn_precomputed_metadata
    return builder.build(common_prefix_len=0, common_attn_metadata=common, **kwargs)


@pytest.mark.parametrize(
    "test_case", GDN_BUILD_TEST_CASES.values(), ids=GDN_BUILD_TEST_CASES.keys()
)
def test_gdn_build_classification(test_case: GDNBuildTestCase):
    """Test that GDN metadata builder classifies requests correctly."""
    builder = _create_gdn_builder(test_case.num_speculative_tokens)
    batch = BatchSpec(seq_lens=test_case.seq_lens, query_lens=test_case.query_lens)
    meta = _build(builder, batch, test_case.num_decode_draft_tokens)

    assert meta.num_decodes == test_case.expected_num_decodes
    assert meta.num_prefills == test_case.expected_num_prefills
    assert meta.num_prefill_tokens == test_case.expected_num_prefill_tokens
    assert meta.num_spec_decodes == test_case.expected_num_spec_decodes


def test_has_initial_state_after_reclassification():
    """After reclassification, num_prefills > 0 so the prefill kernel path
    should compute has_initial_state. For the reclassified request with
    context_lens > 0, the corresponding entry must be True."""
    builder = _create_gdn_builder(num_speculative_tokens=2)
    batch = BatchSpec(seq_lens=[65, 20], query_lens=[1, 3])
    meta = _build(builder, batch, num_decode_draft_tokens=[-1, 2])

    assert meta.num_prefills > 0, "reclassification should produce prefills"
    assert meta.has_initial_state is not None
    # req0 has context_lens = 65 - 1 = 64 > 0, so has_initial_state[0] = True
    assert meta.has_initial_state[0].item() is True


def test_full_cudagraph_spec_metadata_uses_request_count():
    """FULL cudagraph token padding must not pad request-indexed metadata."""
    num_speculative_tokens = 3
    builder = _create_gdn_builder(
        num_speculative_tokens=num_speculative_tokens,
        full_cuda_graph=True,
    )
    batch = BatchSpec(seq_lens=[80, 96], query_lens=[4, 4])
    meta = _build(builder, batch, num_decode_draft_tokens=[3, 3])

    assert meta.num_spec_decodes == batch.batch_size
    assert meta.num_spec_decode_tokens == batch.compute_num_tokens()
    assert meta.spec_state_indices_tensor is not None
    assert meta.spec_state_indices_tensor.shape == (
        batch.batch_size,
        num_speculative_tokens + 1,
    )
    assert meta.spec_sequence_masks is not None
    assert meta.spec_sequence_masks.shape == (batch.batch_size,)
    assert meta.spec_query_start_loc is not None
    assert meta.spec_query_start_loc.shape == (batch.batch_size + 1,)
    assert meta.num_accepted_tokens is not None
    assert meta.num_accepted_tokens.shape == (batch.batch_size,)


def _build_non_spec(
    batch: BatchSpec,
    is_prefilling: list[bool] | None,
    full_cuda_graph: bool = False,
):
    common_attn_metadata = create_common_attn_metadata(
        batch, BLOCK_SIZE, DEVICE
    ).replace(
        is_prefilling=None
        if is_prefilling is None
        else torch.tensor(is_prefilling, dtype=torch.bool)
    )
    builder = _create_gdn_builder(full_cuda_graph=full_cuda_graph)
    return builder, common_attn_metadata, builder.build(0, common_attn_metadata)


@pytest.mark.parametrize(
    ("seq_len", "query_len", "is_prefilling", "num_prefills"),
    [
        pytest.param(1, 1, True, 1, id="first-chunk"),
        pytest.param(65, 1, True, 0, id="resumed-chunk"),
        pytest.param(0, 0, True, 0, id="padding"),
        pytest.param(1, 1, None, 0, id="missing-prefill-flag"),
    ],
)
def test_one_token_chunk_classification(
    seq_len: int,
    query_len: int,
    is_prefilling: bool | None,
    num_prefills: int,
):
    """Only a real first chunk with a prefill flag needs state initialization."""
    _, _, meta = _build_non_spec(
        BatchSpec(seq_lens=[100, seq_len], query_lens=[1, query_len]),
        is_prefilling=None if is_prefilling is None else [False, is_prefilling],
    )

    assert meta.num_prefills == num_prefills
    assert meta.num_decodes == 2 - num_prefills
    assert meta.num_prefill_tokens == num_prefills
    assert meta.num_decode_tokens == 1 + query_len - num_prefills
    if num_prefills:
        assert meta.has_initial_state is not None
        assert meta.has_initial_state.tolist() == [True, False]
    else:
        assert meta.has_initial_state is None


def test_one_token_first_chunk_excludes_padding():
    """Neither padding requests nor padding tokens count as prefill work."""
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[100, 1, 0, 0], query_lens=[1, 1, 0, 0]),
        BLOCK_SIZE,
        DEVICE,
    ).replace(
        is_prefilling=torch.tensor([False, True, False, False], dtype=torch.bool),
        num_actual_tokens=4,
    )
    meta = _create_gdn_builder().build(0, common)

    assert meta.num_decodes == 1
    assert meta.num_prefills == 1
    assert meta.num_decode_tokens == 1
    assert meta.num_prefill_tokens == 1


def test_cudagraph_capture_batch_stays_decode_only():
    """Capture rows have no history, but must still select decode kernels."""
    batch = BatchSpec(seq_lens=[1] * 4, query_lens=[1] * 4)
    builder, common_attn_metadata, _ = _build_non_spec(
        batch, [False] * 4, full_cuda_graph=True
    )
    meta = builder.build_for_cudagraph_capture(common_attn_metadata)

    assert meta.num_prefills == 0
    assert meta.num_decodes == 4
    assert meta.has_initial_state is None
    staged = meta.non_spec_state_indices_tensor
    assert staged is not None
    assert staged.data_ptr() == builder.non_spec_state_indices_tensor.data_ptr()
    torch.testing.assert_close(staged, common_attn_metadata.block_table_tensor[:, 0])


# MRV2 reuses the first kv-cache group's GDN metadata for later groups through
# update_block_table(); the result must match a full build() of each group.
@dataclass
class GroupReuseCase:
    # (kind, seq_len, query_len); kind is "spec", "decode", "prefill" or "pad".
    rows: list[tuple[str, int, int]]
    full_cuda_graph: bool
    num_speculative_tokens: int = 3


GROUP_REUSE_CASES = {
    "spec_decode": GroupReuseCase(
        [("spec", 40, 4), ("spec", 90, 4), ("spec", 17, 4)], True
    ),
    "spec_decode_padded": GroupReuseCase(
        [("spec", 40, 4), ("spec", 90, 4), ("pad", 0, 0), ("pad", 0, 0)], True
    ),
    "spec_decode_dflash": GroupReuseCase(
        [("spec", 60, 8)] * 5 + [("pad", 0, 0)] * 3, True, 7
    ),
    "spec_decode_piecewise": GroupReuseCase(
        [("spec", 40, 4), ("spec", 90, 4)], False
    ),
    "spec_decode_and_prefill": GroupReuseCase(
        [("spec", 40, 4), ("prefill", 30, 30), ("spec", 90, 4), ("prefill", 80, 20)],
        True,
    ),
    "spec_decode_decode_and_prefill": GroupReuseCase(
        [("spec", 40, 4), ("decode", 50, 1), ("prefill", 30, 30), ("pad", 0, 0)],
        False,
    ),
    "decode": GroupReuseCase([("decode", 40, 1), ("decode", 90, 1)], True),
    "decode_padded": GroupReuseCase(
        [("decode", 40, 1), ("decode", 90, 1), ("pad", 0, 0)], True
    ),
    "decode_no_spec_config": GroupReuseCase(
        [("decode", 40, 1), ("decode", 90, 1)], True, 0
    ),
    "decode_and_prefill": GroupReuseCase(
        [("decode", 40, 1), ("decode", 90, 1), ("prefill", 30, 30)], True
    ),
}


def _create_v2_gdn_builders(
    num_builders: int,
    num_speculative_tokens: int,
    full_cuda_graph: bool,
    mamba_cache_mode: str,
) -> list[GDNAttentionMetadataBuilder]:
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B", block_size=BLOCK_SIZE
    )
    if full_cuda_graph:
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    if num_speculative_tokens > 0:
        vllm_config.speculative_config = SpeculativeConfig(
            method="ngram", num_speculative_tokens=num_speculative_tokens
        )
    vllm_config.cache_config.mamba_cache_mode = mamba_cache_mode
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_speculative_tokens,
    )
    # The CPU test host cannot validate a config with MRV2 (it needs Triton).
    with patch.object(VllmConfig, "use_v2_model_runner", property(lambda _: True)):
        builders = [
            GDNAttentionMetadataBuilder(
                kv_cache_spec=mamba_spec,
                layer_names=[f"layer.{i}"],
                vllm_config=vllm_config,
                device=DEVICE,
            )
            for i in range(num_builders)
        ]
    assert all(b.supports_update_block_table for b in builders)
    return builders


def _group_reuse_batch(
    case: GroupReuseCase, num_groups: int
) -> tuple[list[CommonAttentionMetadata], dict]:
    rows = case.rows
    query_lens = torch.tensor([r[2] for r in rows], dtype=torch.int32)
    seq_lens = torch.tensor([r[1] for r in rows], dtype=torch.int32)
    query_start_loc = torch.zeros(len(rows) + 1, dtype=torch.int32)
    query_start_loc[1:] = query_lens.cumsum(0)
    is_pad = torch.tensor([r[0] == "pad" for r in rows])
    num_blocks = 32
    common = []
    for group in range(num_groups):
        generator = torch.Generator().manual_seed(group)
        block_table = torch.randint(
            1, 1000, (len(rows), num_blocks), dtype=torch.int32, generator=generator
        )
        block_table[is_pad] = 0
        num_tokens = int(query_start_loc[-1])
        common.append(
            CommonAttentionMetadata(
                query_start_loc=query_start_loc.to(DEVICE),
                query_start_loc_cpu=query_start_loc,
                seq_lens=seq_lens.to(DEVICE),
                seq_lens_cpu_upper_bound=seq_lens,
                num_reqs=len(rows),
                num_actual_tokens=num_tokens,
                max_query_len=int(query_lens.max()),
                max_seq_len=int(seq_lens.max()),
                block_table_tensor=block_table.to(DEVICE),
                slot_mapping=torch.full((num_tokens,), group, dtype=torch.int64),
                causal=True,
                is_prefilling=torch.tensor([r[0] == "prefill" for r in rows]),
            )
        )
    extra_kwargs = {}
    if case.num_speculative_tokens > 0:
        extra_kwargs = {
            "num_accepted_tokens": torch.arange(1, len(rows) + 1, dtype=torch.int32),
            "num_decode_draft_tokens_cpu": torch.tensor(
                [r[2] - 1 if r[0] == "spec" else -1 for r in rows],
                dtype=torch.int32,
            ),
        }
    return common, extra_kwargs


def _assert_same(actual, expected, name: str) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), name
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=name)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), name
        for key in expected:
            _assert_same(actual[key], expected[key], f"{name}[{key}]")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), name
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_same(a, e, f"{name}[{i}]")
    else:
        assert actual == expected, f"{name}: {actual} != {expected}"


def _assert_metadata_equal(
    actual: GDNAttentionMetadata, expected: GDNAttentionMetadata
) -> None:
    for field in fields(GDNAttentionMetadata):
        _assert_same(
            getattr(actual, field.name), getattr(expected, field.name), field.name
        )


def _set_aligned_state_indices(builder, common: CommonAttentionMetadata) -> None:
    # MRV2 computes these for every group in one fused launch.
    builder.mamba_aligned_state_indices = mamba_get_block_table_tensor(
        common.block_table_tensor, common.seq_lens, builder.kv_cache_spec, "align"
    )


@pytest.mark.parametrize("mamba_cache_mode", ["none", "align"])
@pytest.mark.parametrize(
    "case", GROUP_REUSE_CASES.values(), ids=GROUP_REUSE_CASES.keys()
)
def test_update_block_table_matches_build(case: GroupReuseCase, mamba_cache_mode):
    num_groups = 3
    reused = _create_v2_gdn_builders(
        num_groups, case.num_speculative_tokens, case.full_cuda_graph, mamba_cache_mode
    )
    reference = _create_v2_gdn_builders(
        num_groups, case.num_speculative_tokens, case.full_cuda_graph, mamba_cache_mode
    )
    common, extra_kwargs = _group_reuse_batch(case, num_groups)
    if mamba_cache_mode == "align":
        for builders in (reused, reference):
            for builder, group_common in zip(builders, common):
                _set_aligned_state_indices(builder, group_common)

    first = reused[0].build(0, common[0], **extra_kwargs)
    _assert_metadata_equal(first, reference[0].build(0, common[0], **extra_kwargs))
    for group in range(1, num_groups):
        actual = reused[group].update_block_table(
            first, common[group].block_table_tensor, common[group].slot_mapping
        )
        expected = reference[group].build(0, common[group], **extra_kwargs)
        _assert_metadata_equal(actual, expected)
        # Only the state indices are per group; FULL-graph staging writes them
        # into this group's buffers and shares the first group's batch buffers.
        for name in ("spec_state_indices_tensor", "non_spec_state_indices_tensor"):
            buffer = getattr(reused[group], name)
            staged = getattr(actual, name)
            expected_staged = getattr(expected, name)
            is_staged = (
                expected_staged is not None
                and expected_staged.data_ptr()
                == getattr(reference[group], name).data_ptr()
            )
            assert (staged is not None and staged.data_ptr() == buffer.data_ptr()) == (
                is_staged
            ), name
        for name in (
            "spec_query_start_loc",
            "non_spec_query_start_loc",
            "spec_sequence_masks",
            "spec_token_indx",
            "non_spec_token_indx",
            "num_accepted_tokens",
            "has_initial_state",
            "chunk_indices",
        ):
            assert getattr(actual, name) is getattr(first, name), name


def test_aligned_state_indices_match_gather():
    """Precomputed align-mode state indices must match the per-group gather."""
    case = GROUP_REUSE_CASES["spec_decode_and_prefill"]
    builders = _create_v2_gdn_builders(2, 3, True, "align")
    common, extra_kwargs = _group_reuse_batch(case, 1)
    _set_aligned_state_indices(builders[0], common[0])
    _assert_metadata_equal(
        builders[0].build(0, common[0], **extra_kwargs),
        builders[1].build(0, common[0], **extra_kwargs),
    )


@pytest.mark.parametrize("for_cudagraph_capture", [False, True])
@pytest.mark.parametrize("case_name", ["spec_decode_padded", "decode_and_prefill"])
def test_build_attn_metadata_reuses_gdn_groups(for_cudagraph_capture, case_name):
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend
    from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridAttnMetadata
    from vllm.v1.worker.utils import AttentionGroup

    case = GROUP_REUSE_CASES[case_name]
    if for_cudagraph_capture:
        case = GroupReuseCase([("spec", 40, 4), ("spec", 90, 4)], True)
    num_groups = 3
    builders = _create_v2_gdn_builders(
        num_groups, case.num_speculative_tokens, case.full_cuda_graph, "none"
    )
    reference = _create_v2_gdn_builders(
        num_groups, case.num_speculative_tokens, case.full_cuda_graph, "none"
    )
    common, extra_kwargs = _group_reuse_batch(case, num_groups)
    attn_groups = []
    for group, builder in enumerate(builders):
        attn_group = AttentionGroup(
            GDNAttentionBackend, builder.layer_names, builder.kv_cache_spec, group
        )
        attn_group.metadata_builders = [builder]
        attn_groups.append([attn_group])
    m = common[0]
    attn_metadata = build_attn_metadata(
        attn_groups=attn_groups,
        num_reqs=m.num_reqs,
        num_tokens=m.num_actual_tokens,
        query_start_loc_gpu=m.query_start_loc,
        query_start_loc_cpu=m.query_start_loc_cpu,
        max_query_len=m.max_query_len,
        seq_lens=m.seq_lens,
        max_seq_len=m.max_seq_len,
        block_tables=[c.block_table_tensor for c in common],
        slot_mappings=torch.stack([c.slot_mapping for c in common]),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[None] * num_groups),
        seq_lens_cpu_upper_bound=m.seq_lens_cpu_upper_bound,
        model_specific_attn_metadata=MambaHybridAttnMetadata(
            is_prefilling=m.is_prefilling,
            num_accepted_tokens=extra_kwargs.get("num_accepted_tokens"),
            num_decode_draft_tokens_cpu=extra_kwargs.get(
                "num_decode_draft_tokens_cpu"
            ),
        ),
        for_cudagraph_capture=for_cudagraph_capture,
    )
    first = attn_metadata["layer.0"]
    for group in range(num_groups):
        actual = attn_metadata[f"layer.{group}"]
        if for_cudagraph_capture:
            expected = reference[group].build_for_cudagraph_capture(common[group])
        else:
            expected = reference[group].build(0, common[group], **extra_kwargs)
        _assert_metadata_equal(actual, expected)
        # Later groups read the first group's batch-level buffers, also in the
        # capture-time metadata a FULL graph would bake in.
        assert actual.non_spec_query_start_loc is first.non_spec_query_start_loc
        assert actual.spec_query_start_loc is first.spec_query_start_loc


def test_update_block_table_only_for_mrv2_and_gdn_build():
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B", block_size=BLOCK_SIZE
    )
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE, shapes=((16, 64),), dtypes=(torch.float16,)
    )

    class CustomBuild(GDNAttentionMetadataBuilder):
        def build(self, *args, **kwargs):  # type: ignore[override]
            return super().build(*args, **kwargs)

    for use_v2 in (False, True):
        with patch.object(
            VllmConfig, "use_v2_model_runner", property(lambda _: use_v2)
        ):
            gdn = GDNAttentionMetadataBuilder(mamba_spec, ["l"], vllm_config, DEVICE)
            custom = CustomBuild(mamba_spec, ["l"], vllm_config, DEVICE)
        # MRV1 reuses metadata only at replay, which FULL graphs cannot follow.
        assert gdn.supports_update_block_table is use_v2
        assert not custom.supports_update_block_table
