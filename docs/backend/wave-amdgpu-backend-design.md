# Wave AMDGPU Backend Design

## Goal

Add `/home/vano/7/7` as a Triton AMDGPU backend that lowers Triton kernels into Wave IR, then uses the Wave compiler to emit AMDGPU HSACO.

The backend must preserve Wave's programming model:

- subgroup/wavefront execution is explicit;
- lane-varying values are first-class;
- masks remain explicit;
- address calculations remain symbolic until WaveAMDMachine lowering.

## Key Constraint

Normal Triton AMDGPU lowering converts TTIR to TTGIR, assigns backend layouts, rewrites memory operations, then lowers to LLVM.

That is too late for Wave.

Wave wants to own:

- wave/subgroup decomposition;
- per-lane indexing;
- mask structure;
- pointer offset algebra;
- mapping from program IDs and lanes to memory coordinates.

Therefore the Wave backend should intercept before Triton's normal TTGIR layout pipeline, not at LLVM IR or AMDGCN.

Concrete boundary:

```text
after ASTSource.make_ir / TTIR cleanup
before passes.ttir.add_convert_to_ttgpuir
```

The Wave backend should not call AMD `make_ttgir()` as its main lowering path.

## Triton Boundary

Relevant Triton files:

- `/home/vano/triton/python/triton/compiler/compiler.py`
- `/home/vano/triton/python/triton/backends/compiler.py`
- `/home/vano/triton/python/triton/backends/__init__.py`
- `/home/vano/triton/third_party/amd/backend/compiler.py`
- `/home/vano/triton/third_party/amd/backend/driver.py`

Triton backend stages are an ordered map:

```text
stage_name -> function(module_or_text, metadata) -> module_or_text_or_bytes
```

HIP currently uses:

```text
ttir -> ttgir -> llir -> amdgcn -> hsaco
```

Wave should use:

```text
ttir -> wave -> amdgcn -> hsaco
```

or, if a small amount of TTIR normalization is still useful:

```text
ttir_cleanup -> wave -> amdgcn -> hsaco
```

It should not consume fully lowered AMD TTGIR as the primary path.

## Backend Shape

Implement the backend in this Triton branch.

Backend root:

```text
/home/vano/triton/third_party/wave_amd
```

Wave compiler submodule:

```text
/home/vano/triton/third_party/wave_amd/wave
```

Backend name:

```text
wave_amd
```

Target:

```python
GPUTarget("wave_amd", "gfx1100", 32)
```

Do not register as `hip`; Triton requires exactly one backend per active target.

Build/install integration:

- add `wave_amd` to Triton's in-tree backend list;
- install `triton.backends.wave_amd`;
- build native Wave/Triton glue from `third_party/wave_amd/CMakeLists.txt`;
- use the Wave submodule for dialects, passes, and AMDGPU emission.

Python backend responsibilities:

- implement `BaseBackend`;
- set `binary_ext = "hsaco"`;
- parse AMD target/options;
- register Wave dialects in `load_dialects`;
- lower TTIR to Wave MLIR;
- call Wave compiler to produce AMDGCN/HSACO;
- populate Triton launch metadata.

Driver responsibilities:

- initially reuse HIP runtime loading and launching;
- return a `wave_amd` target;
- keep final binary format as HSACO;
- keep kernel argument ABI compatible with Triton's HIP launcher.

## Wave Compiler Boundary

Relevant Wave files:

- `/home/vano/triton/third_party/wave_amd/wave/include/mlir/Target/Wave/AMDGPU.h`
- `/home/vano/triton/third_party/wave_amd/wave/lib/Target/Wave/pipelines/pipelines.mlir`
- `/home/vano/triton/third_party/wave_amd/wave/lib/Target/Wave/WaveCompileKernels.cpp`
- `/home/vano/triton/third_party/wave_amd/wave/python/WaveExtensionNanobind.cpp`
- `/home/vano/triton/third_party/wave_amd/wave/python/mlir/dialects/wave_dsl.py`

Wave accepts Wave/WaveAMD MLIR marked with:

```text
wave.kernel
waveamdmachine.target = "amdgcn-amd-amdhsa--gfx..."
```

Wave pipeline:

```text
waveamd_backend_lower
waveamd-machine-schedule
waveamd_backend_finish
wave-compile-kernels
```

Emission options:

- initial: call submodule-built `wave-translate --wave-to-amdgpu-asm`, then assemble/link;
- target: call `translateWaveToAMDGPU()` or `assembleWaveAMDGPUKernels()` in-process through Triton native glue.

## TTIR To Wave Lowering

The important bridge is TTIR to Wave, not TTGIR to Wave.

Required mappings:

- `tl.program_id(axis)` -> `wave.workgroup_id axis`;
- implicit lane id -> `wave.workitem_id` / `wave.lane_id`;
- scalar values -> uniform Wave values;
- block tensors -> `!wave.simd<T, W>` or tuples/fragments;
- masks -> `!wave.mask<W>` and `wave.where`;
- pointer bases -> `!wave.ptr`;
- pointer offsets -> `wave.index_expr`;
- loads/stores -> Wave memory ops with explicit mask and symbolic offset;
- `tl.dot` / matmul forms -> `waveamd.mma` or fragment ops;
- shared memory -> Wave LDS constructs;
- barriers/waits -> Wave/WaveAMD synchronization ops.

## Memory Tokens

Wave memory ordering is explicit SSA.

Relevant Wave ops:

- `wave.token`;
- `wave.after`;
- `wave.join`;
- `wave.wait`;
- `wave.barrier`;
- `wave.load`;
- `wave.store`;
- `waveamd.dma_load_lds`.

`wave.load`, `wave.store`, and DMA ops consume an optional dependency token and return a new token. `wave.barrier` consumes dependencies and returns a token for operations after the barrier.

Triton TTIR does not carry this token graph. Normal `tt.load` / `tt.store` encode side effects, masks, volatility, cache policy, and eviction policy. Atomics carry memory semantic and scope. `tl.debug_barrier()` becomes a Triton barrier op.

The bridge must synthesize Wave tokens while lowering TTIR.

Initial policy:

```text
one token state per function block and memory space
```

At function entry:

```text
global_token = wave.token
shared_token = wave.token
```

For each lowered memory op:

- choose token state from address space;
- pass it as the Wave dependency when ordering is required;
- update that token state from the op result token;
- for operations touching multiple spaces, update with `wave.join`.

Conservative first version:

- chain stores, atomics, volatile loads, DMA, and barriers;
- chain ordinary loads only when needed for a later memory-order dependency;
- rely on SSA value dependencies for load-result consumers;
- emit `wave.wait` when Triton semantics require completion before non-memory use;
- emit `wave.barrier` for `tl.debug_barrier()` with joined outstanding dependencies.

This preserves correctness without forcing all independent loads into one serial chain.

Later refinements:

- split token state by alias class, not just memory space;
- avoid chaining read-only global loads;
- model atomic `sem` / `scope` explicitly;
- thread loop-carried tokens through `scf.for` / `scf.while`;
- merge branch tokens at control-flow joins with `wave.join`.

WaveAMDMachine lowering preserves these tokens as scheduler dependencies. `waveamd-insert-ticket-waits` then materializes the required `s_waitcnt` / `s_waitcnt_vscnt` instructions.

Address lowering rule:

```text
Do not materialize pointer arithmetic into ordinary integer IR unless Wave requires it.
```

Keep expressions symbolic so Wave can reason about:

- lane modulo/decomposition;
- wave id decomposition;
- tile coordinates;
- stride expressions;
- packed memory access patterns;
- range assumptions.

## Metadata Contract

The backend must fill the metadata consumed by `CompiledKernel` and the HIP-style launcher:

- `name`;
- `num_warps`;
- `num_ctas`;
- `shared`;
- `warp_size`;
- `global_scratch_size`;
- `global_scratch_align`;
- `profile_scratch_size`;
- `profile_scratch_align`;
- `tensordesc_meta` if tensor descriptors are supported.

If Wave's kernel ABI diverges from Triton HIP's flattened ABI, fix the ABI before broadening kernel coverage.

## Integration Points

Preferred:

```text
in-tree wave_amd backend: TTIR -> Wave -> HSACO
```

Useful for experiments:

```text
triton.knobs.runtime.add_stages_inspection_hook
```

Use the hook only to prove the pipeline and metadata contract. Do not base the design on it.

Native plugin option:

```text
TRITON_PLUGIN_PATHS + tritonGetPluginInfo
```

Not the main path. Use only for experiments that should stay outside the branch.

## Non-Goals

- Do not implement the main path as an external Triton plugin.
- Do not replace Triton's in-tree HIP backend.
- Do not target LLVM IR as the primary Wave interface.
- Do not lower through Triton AMD TTGIR layouts unless used only for comparison.
- Do not support the full Triton language surface in the first version.

## First Supported Kernels

Start with kernels that exercise the real boundary:

1. elementwise masked load/store;
2. strided copy with non-trivial symbolic offsets;
3. tiled matmul mapped to Wave/WaveAMD fragments.

Success criteria:

- Wave MLIR preserves symbolic address expressions;
- generated HSACO loads through Triton's runtime;
- outputs match HIP backend reference;
- dumped artifacts include TTIR, Wave MLIR, AMDGCN, HSACO metadata.

## Open Decisions

- Whether the bridge should consume raw TTIR or TTIR after a small common cleanup pass.
- Exact representation of Triton block tensors in Wave types.
- Token synthesis policy for loops, branches, atomics, and alias classes.
- ABI compatibility for tensor descriptors and scratch allocations.
- Whether to expose Wave as a Python-only backend first or add native Triton bindings immediately.
