# Wave AMD GEMM Pipeline Design

## Status

Design for the Wave AMD Triton backend GEMM path. Partially implemented.

A canonical f16 GEMM now compiles and runs end to end through the Wave AMD
backend with correct numerics, including `num_warps=4` and a K loop, verified on
gfx1100 (RDNA3, WMMA). The Triton-tree C++ pass infrastructure the design calls
for exists, and three TTGIR preparation passes have landed.

### Implementation status (2026-06-09)

C++ TTGIR preparation passes (Triton tree, `backend/passes/`, registered into
`triton-opt` and the `TritonWaveAMD` plugin):

- `tritonwaveamd-convert-to-ttgpuir`: Wave-owned Triton-to-TritonGPU conversion.
  Currently byte-identical to the base pass (reuses the extracted
  `mlir::triton::convertToTritonGPU` helper); the pipeline calls it instead of
  `passes.ttir.add_convert_to_ttgpuir`. The seam exists; the layouts have not
  diverged yet.
- `tritonwaveamd-accelerate-matmul`: Wave-owned matrix-core dot legalization.
  Currently byte-identical to `amd.accelerate_matmul` (reuses the extracted
  `mlir::triton::amdgpu::accelerateMatmul` helper); the pipeline calls it
  instead of the AMD pass.
- `tritonwaveamd-legalize-dots`: rejects MFMA with a diagnostic; attaches
  `waveamd.dot.{instr_kind,a_op_idx,b_op_idx}` and `waveamd.dot.role` on operand
  convert_layouts. **Consumed** by the bridge.
- `tritonwaveamd-plan-gemm-schedule`: attaches GEMM tiling + LDS plan attrs.
  **Latent** -- see below, LDS staging was removed.
- `tritonwaveamd-plan-buffer-descriptors`: attaches `waveamd.buffer.range_bytes`.
  Bridge consumes it, but the static-affine buffer-ops path is currently
  unexercised by the test kernels (they take the symbolic dot-layout path).

Migration phase scorecard (details inline in Migration Plan below):

- Phase 2 (architecture policy): DONE and consumed.
- Phase 5 (dot planning, dot slice): DONE -- bridge reads `instr_kind`/`role`.
- Phase 6 (LDS/schedule): the dot-fragment LDS staging turned out to be an
  identity round-trip (no data movement); it was REMOVED, kernels reserve no
  shared memory, and the schedule pass is now a latent producer.
- Phase 7 (buffer descriptors): pass + consumer landed; path latent.
- Phase 8 (delete bridge analysis): dot-path type-string parsing deleted; layout
  and epilogue analysis still live.
- Phase 3 (replace `add_convert_to_ttgpuir`): producer SEAM owned. The pipeline
  no longer calls the base conversion or `amd.accelerate_matmul`; both are
  Wave-owned passes (`tritonwaveamd-convert-to-ttgpuir`,
  `tritonwaveamd-accelerate-matmul`). They are still byte-identical reuses, so
  the emitted layouts are still TritonGPU encodings, not Wave-native contracts;
  diverging them is the remaining Phase 3 work.
- Phase 4 (layout exprs to C++): NOT STARTED; depends on the Phase 3 layout
  divergence above.

Corrections to the original assumptions below:

- The bridge does NOT reconstruct blocked layouts for the GEMM path; it builds
  symbolic dot-layouts and `wave.index_expr` directly. Phase 4 is therefore
  "move the symbolic layout construction from Python to C++", not "remove
  blocked-layout reconstruction".
- LDS staging is not a load-bearing step today; the WMMA fragments are already in
  a usable layout without it.
- Structural attribute APIs are sufficient through existing Python MLIR bindings
  (`op.get_int_attr` / `op.get_str_attr` return None when absent); no C API
  additions were needed for the passes landed so far.

## Goals

- Reuse base Triton passes wherever they preserve Wave's execution model.
- Treat Triton AMD TTGIR passes as reusable policy or test oracles when their IR
  rewrites commit to TritonGPU layouts.
- Reimplement non-reusable Triton AMD behavior as Wave-owned C++ TTIR/TTGIR
  passes in the Triton tree, not in the Wave submodule.
- Keep Python limited to the TTGIR-to-Wave bridge in `lowering.py`.
- Lower layout math into `wave.index_expr`.
- Keep TTGIR-to-Wave lowering as a dumb bridge: no layout discovery, no GEMM
  scheduling, no implicit dot role analysis, no pointer range inference.
- Preserve a pipeline shape close enough to base Triton that canonical GEMM
  tutorials, autotune knobs, and lit tests remain useful inputs.

## Non-goals

- Do not lower through Triton AMDGPU LLVM dialect ops and translate them back to
  Wave.
- Do not make `lowering.py` a second optimizer.
- Do not encode Wave layout decisions only as printed TTGIR type strings.
- Do not require MFMA support before the WMMA path has a stable preparation
  pipeline. MFMA should enter through the same contracts.

## Pipeline Shape

```text
Python Triton AST
  -> TTIR
  -> reusable TTIR cleanup
  -> Wave C++ TTIR preparation
  -> Wave C++ TTIR-to-TTGIR conversion
  -> Wave C++ TTGIR GEMM preparation
  -> dumb Python TTGIR-to-Wave bridge
  -> Wave dialect optimization and WaveAMD machine lowering
  -> AMDGCN
  -> HSACO
```

The bridge should see a Wave-ready TTGIR module. It may map operations, copy
attributes, and materialize prepared expressions. It should not infer missing
structure from arbitrary TTIR/TTGIR graphs.

Current divergence: the "Wave C++ TTIR-to-TTGIR conversion" step exists as a
Wave-owned pass (`tritonwaveamd-convert-to-ttgpuir`) but still emits TritonGPU
layouts byte-identically to the base conversion, so it is not yet a Wave-native
layout contract. The pipeline runs that Wave conversion plus reused TTGIR
cleanup, then the Wave-owned `tritonwaveamd-accelerate-matmul`, then the three
Wave C++ TTGIR preparation passes. "Wave C++ TTIR preparation" is not a separate
step yet.

## Repository Ownership

TTIR and TTGIR preparation passes belong to the Triton-side Wave AMD backend
tree, for example under `third_party/wave_amd/backend` and Triton pass
registration code. They are C++ MLIR passes that transform Triton dialect IR and
depend on Triton pass pipeline decisions, so they must not live in
`third_party/wave_amd/wave`.

The Wave submodule starts at Wave dialect IR. Its responsibilities are Wave
dialect canonicalization, `wave.index_expr` simplification, pointer/address
normalization, WaveAMD machine lowering, hazards, waits, scheduling, and target
emission.

The only Python compiler stage in this plan is the bridge in `lowering.py`. It
constructs Wave dialect IR from already-prepared TTGIR and performs no compiler
analysis.

## Reuse Policy

### Reuse As-is

These passes operate before GPU layout decisions or are generic cleanup:

- `common.inliner`
- `common.canonicalizer`
- `common.cse`
- `common.symbol_dce`
- `ttir.combine`
- `ttir.reorder_broadcast`
- `ttir.triton_licm`
- `ttir.loop_unroll`

They belong in `pipeline.py` before Wave-specific preparation.

### Reuse With Constraints

These passes can be considered only after the new Wave TTIR-to-TTGIR conversion
defines Wave-compatible TTGIR contracts:

- `ttgpuir.coalesce`: useful access grouping, but output layouts are not Wave
  layouts.
- `ttgpuir.f32_dot_tc`: useful dot decomposition semantics.
- `ttgpuir.remove_layout_conversions`: safe only after Wave marks required
  conversion boundaries.

`ttir.convert_to_ttgpuir` is not reused. Wave needs a C++ TTIR-to-TTGIR
conversion written from scratch so the first TTGIR layouts are already
Wave-compatible instead of TritonGPU layout approximations.

### Adapt As Policy

These Triton AMD passes contain valuable decisions, but their rewrites target
TritonGPU/AMDGPU IR:

- `amd.accelerate_matmul`: reuse intrinsic legality, architecture policy, and
  shape selection. Reimplement Wave layout materialization.
- `amd.optimize_dot_operands`: reuse operand-role intent. Reimplement LDS and
  fragment layout choices.
- `amd.schedule_loops`: reuse software pipeline concepts. Reimplement schedule
  representation for Wave tokens and LDS.
- `amd.pipeline`: reuse prologue/main/epilogue structure as an oracle.
  Reimplement expansion with Wave memory, barrier, wait, and token semantics.
- `amd.canonicalize_pointers`: reuse pointer-base/vector-offset split. Lower the
  result into `wave.index_expr` and Wave buffer descriptors.

### Avoid Direct Reuse

These passes commit to AMDGPU SIMT or LLVM-facing details:

- `amd.convert_to_buffer_ops`
- `amd.optimize_descriptor_encoding`
- `amd.convert_to_tensor_ops`
- `amd.block_pingpong`
- `amd.in_thread_transpose`
- `amd.coalesce_async_copy`
- `amd.update_async_wait_count`

Wave should reimplement their relevant behavior either in Triton-tree Wave
TTIR/TTGIR preparation or, after the bridge, in downstream Wave dialect passes.
Do not add TTIR/TTGIR passes to the Wave submodule.

## Wave TTIR Preparation

Wave TTIR preparation should run after reusable TTIR cleanup and before the
Wave-owned TTIR-to-TTGIR conversion. These C++ passes live in the Triton tree.

Responsibilities:

- Canonicalize GEMM-like `tt.dot` patterns and simple fused epilogues.
- Preserve dot accumulators through identity epilogues before fragment lowering.
- Normalize block pointer and raw pointer arithmetic into a form that can become
  symbolic Wave index expressions.
- Split uniform base pointers from lane-varying offsets.
- Attach stable metadata for masks, program ids, ranges, and static footprints.
- Reject unsupported source constructs early with diagnostics that point to the
  unsupported Triton operation, not the later Wave bridge.

Output contract:

- TTIR remains valid Triton IR.
- Values that will become Wave indices have explicit symbolic components.
- Pointer arithmetic has a recoverable base plus symbolic offset.
- Dot operands are still semantic Triton values; no Wave fragment ops are emitted
  yet.

## Wave TTIR-to-TTGIR Conversion

Wave does not use `add_convert_to_ttgpuir`. The conversion is rewritten from
scratch as a Triton-tree C++ pass.

Responsibilities:

- Create TTGIR with Wave-compatible layout contracts from the start.
- Preserve Triton `tt.load`, `tt.store`, `tt.dot`, masks, program ids, and loop
  structure in forms the Wave TTGIR preparation passes can consume.
- Attach explicit layout anchors for tensors that will become `wave.index_expr`
  calculations.
- Avoid introducing TritonGPU layouts that later require recovery or string
  parsing in the Python bridge.

Output contract:

- The module is TTGIR, but its layouts are Wave contracts, not temporary
  TritonGPU encodings.
- All required layout and pointer facts are represented structurally.
- Later Wave TTGIR passes can legalize GEMM without reconstructing frontend
  intent.

## Wave TTGIR GEMM Preparation

Wave TTGIR preparation owns layout, dot, and pipeline decisions. These C++
passes live in the Triton tree and should be the primary replacement for
analysis currently embedded in `lowering.py`.

Responsibilities:

- Legalize each `tt.dot` against the Wave architecture policy.
- Select the matrix instruction kind, for example
  `wmma.f32.16x16x16.f16`.
- Assign operand roles A and B without relying on type-string parsing in the
  bridge.
- Assign result fragment layout and accumulator ownership.
- Materialize Wave symbolic layout maps for:
  - lane id
  - wave id
  - CTA id
  - tile coordinates
  - K-step coordinates
  - LDS slot offsets
  - global memory offsets
- Plan software-pipelined K iteration:
  - number of K steps
  - stage number per step
  - LDS slots
  - barrier placement
  - token dependencies
  - prologue/main/epilogue shape
- Decide LDS footprint and attach a kernel-level requirement.
- Decide buffer descriptor ranges for static footprints.
- Preserve required `ttg.convert_layout` anchors or replace them with explicit
  Wave preparation attributes.

Output contract:

- Every layout expression that affects memory, masks, or fragment placement has
  a Wave-compatible symbolic representation.
- Every dot has an explicit instruction kind, operand roles, fragment geometry,
  and accumulator layout.
- Every staged load/store has explicit LDS slot, stage, and token dependency
  metadata.
- Kernel-level shared memory and buffer range requirements are explicit attrs.

## TTGIR-to-Wave Bridge Contract

The bridge lowers prepared IR mechanically.

Allowed work:

- Map `tt.func` to a Wave function and copy prepared kernel attrs.
- Map `tt.get_program_id`, `tt.make_range`, masks, and pointer offsets to
  prepared `wave.index_expr` values.
- Map prepared loads/stores to `wave.load`, `wave.store`,
  `waveamd.make_buffer`, or `waveamd.dma_load_lds`.
- Map prepared dot fragments to `waveamd.fragment_pack`,
  `waveamd.fragment_unpack`, and `waveamd.mma`.
- Emit prepared `wave.barrier`, `wave.after`, `wave.join`, and `wave.wait`
  structure.
- Assert that required preparation attrs exist.

Forbidden work:

- Deriving blocked layouts from tensor shapes.
- Parsing TTGIR type strings to discover dot roles or architecture policy.
- Computing GEMM schedules.
- Assigning LDS slots.
- Inferring buffer descriptor ranges.
- Folding epilogues.
- Proving ownership of output tiles.
- Reconstructing affine index maps from arbitrary arithmetic.

If the bridge cannot lower an op without analysis, the preparation pipeline is
missing a contract.

## `wave.index_expr` Contract

Layout math must lower into `wave.index_expr`, not Python-side arithmetic hidden
inside the bridge.

Preparation should build symbolic expressions with named bindings for:

- `lane`: workitem lane inside the wave.
- `wave`: wave id inside the CTA.
- `cta_m`, `cta_n`, `cta_k`: CTA tile coordinates.
- `tile_m`, `tile_n`: matrix instruction tile coordinates.
- `k_step`: current K tile.
- `stage`: software pipeline stage.
- `lds_slot`: chosen LDS slot.

The bridge may instantiate these expressions with Wave SSA values. Downstream
Wave passes then own simplification, pointer offset normalization, loop stride
extraction, address bucketization, and machine lowering.

## Data Model

Wave preparation should stop passing structural compiler facts through strings.
Use typed attributes emitted by C++ passes and consumed mechanically by the
Python bridge.

Suggested concepts:

- `WaveLayoutExpr`: symbolic expression plus bindings and result element type.
- `WavePointerPlan`: base pointer, static footprint, dynamic symbolic offset,
  mask expression, cache/eviction policy.
- `WaveDotPlan`: instruction kind, operand roles, fragment geometry, accumulator
  layout, owner predicate.
- `WaveGemmSchedule`: K-step count, stage assignment, LDS plan, token plan.
- `WaveKernelResources`: LDS bytes, buffer descriptor ranges, required wave
  size, required matrix instruction features.

Existing `gemm_pipeline.py` classes can seed this model, but they should become
the output of preparation rather than bridge-local analysis.

## Architecture Policy

Architecture policy should be a first-class preparation step:

- RDNA WMMA paths may select supported `waveamd.mma` WMMA forms.
- CDNA MFMA paths should be rejected until MFMA fragment layout support is wired
  through the same preparation contracts.
- Policy should reuse Triton AMD intrinsic legality tables where practical:
  `TargetFeatures`, `MfmaIntrinsic`, and `WmmaIntrinsic` are the relevant
  sources of truth.
- Unsupported architecture, dtype, K width, or tile shape failures should happen
  before the bridge.

The same contract should later allow MFMA by adding new instruction kinds and
fragment layouts, not by adding MFMA-specific analysis to the bridge.

## Testing Strategy

Use base Triton and AMD tests as oracles, then assert Wave-specific contracts.

Frontend and compile tests:

- Canonical Triton tutorial GEMM shapes.
- HIP autotune knobs: `num_warps`, `num_stages`,
  `matrix_instr_nonkdim`, `kpack`.
- Dynamic masks and boundary checks.
- Fused identity epilogues and simple non-identity epilogues.

Preparation tests:

- Dot legalization produces explicit instruction kind and roles.
- Layout maps lower to expected symbolic expressions.
- K-loop schedule records stages, LDS slots, token dependencies, and LDS bytes.
- Unsupported MFMA or unsupported shapes fail before bridge lowering.
- Required `ttg.convert_layout` anchors are preserved or replaced by explicit
  Wave attrs.

Bridge tests:

- Prepared index metadata becomes `wave.index_expr`.
- Prepared fragments become `waveamd.fragment_*` and `waveamd.mma`.
- Prepared LDS plans become `wave.lds_base`, tokenized loads/stores or DMA, and
  barriers.
- Bridge rejects missing preparation attrs.

Wave pipeline tests:

- `wave-simplify-index-exprs`
- `wave-normalize-pointer-offsets`
- `wave-combine-pointer-offsets`
- `wave-extract-loop-strides`
- `waveamd-to-machine`
- hazard and wait insertion passes

Useful Triton AMD oracle tests:

- `test/TritonGPU/amd/accelerate-amd-matmul-*.mlir`
- `test/TritonGPU/amd/amd-optimize-dot-operands.mlir`
- `test/TritonGPU/amd/amd-pipeline*.mlir`
- `test/TritonGPU/amd/amd-convert-buffer-ops*.mlir`
- `test/TritonGPU/amd/amd-block-pingpong*.mlir`

Useful Wave tests:

- `third_party/wave_amd/wave/test/Target/Wave/waveamdmachine-index-expr.mlir`
- `third_party/wave_amd/wave/test/Target/Wave/waveamdmachine-buffer.mlir`
- `third_party/wave_amd/wave/test/Target/Wave/waveamdmachine-gfx950-dma-matmul.mlir`
- `third_party/wave_amd/wave/test/Integration/wave_wmma*.mlir`
- `third_party/wave_amd/wave/test/Integration/wave_mfma*.mlir`

## Migration Plan

1. Freeze the bridge contract. [PARTIAL]
   - Add bridge tests that require prepared attrs for GEMM lowering.
   - Keep current behavior only behind temporary compatibility helpers.
   - Status: bridge/lit/e2e tests exist; the bridge does not yet fail fast when
     preparation has not run (it falls back to local analysis for layouts).

2. Move architecture policy out of `lowering.py`. [DONE]
   - Add a Wave TTGIR legalization step.
   - Reuse Triton AMD target feature and intrinsic legality logic where possible.
   - Fix option plumbing so `matrix_instr_nonkdim` has one default.
   - Status: `tritonwaveamd-legalize-dots` rejects MFMA in TTGIR; the bridge no
     longer parses encodings for dot policy. Option plumbing fixed:
     `matrix_instr_nonkdim` had a duplicate field with conflicting defaults
     (16 advertised, 0 in effect); collapsed to a single default of 0.

3. Replace `add_convert_to_ttgpuir`. [PARTIAL: producer owned]
   - Add a Triton-tree C++ TTIR-to-TTGIR conversion for Wave.
   - Emit Wave-compatible layout contracts directly.
   - Remove dependence on TritonGPU layout approximations for the Wave path.
   - Status: the Wave pipeline no longer calls the base conversion or
     `amd.accelerate_matmul`; both are Wave-owned passes
     (`tritonwaveamd-convert-to-ttgpuir`, `tritonwaveamd-accelerate-matmul`),
     each a byte-identical reuse of the extracted base body
     (`mlir::triton::convertToTritonGPU`, `mlir::triton::amdgpu::accelerateMatmul`).
     The open question below is now half-answered in practice: Wave reuses the
     AMD intrinsic selection wholesale by calling its legalization. What remains
     is emitting Wave-native layouts instead of TritonGPU encodings -- the
     reason the passes are still "byte-identical" rather than "diverged".

4. Move layout expression construction out of `lowering.py`. [NOT STARTED]
   - Introduce symbolic layout/index metadata in TTIR/TTGIR preparation.
   - Lower those expressions to `wave.index_expr` in the bridge.
   - Remove bridge-side blocked layout reconstruction.
   - Status: the bridge already builds symbolic dot-layouts / `wave.index_expr`
     (no blocked reconstruction on the GEMM path), but it CONSTRUCTS them in
     Python. The work is moving that construction to C++. Depends on phase 3.

5. Move dot planning out of `lowering.py`. [DONE, dot slice]
   - Emit explicit dot plans with operand roles, fragment geometry, and
     accumulator layout.
   - Bridge maps plans to fragment pack/unpack and `waveamd.mma`.
   - Status: bridge reads `waveamd.dot.instr_kind` and `waveamd.dot.role` from
     legalize-dots. Fragment geometry / accumulator layout are still derived in
     `_dot_config` in the bridge.

6. Move LDS and software pipeline planning out of `lowering.py`. [SUPERSEDED]
   - Turn `WaveGemmSchedule` into preparation output.
   - Attach stage, LDS slot, token, and resource attrs before bridge lowering.
   - Bridge emits only the prepared Wave ops.
   - Status: the dot-fragment LDS staging was a verified identity round-trip and
     was removed; GEMM kernels reserve no shared memory. The schedule pass
     attaches the plan but is currently a latent producer. Revisit if a real
     layout-changing LDS staging is needed.

7. Move buffer descriptor planning out of `lowering.py`. [DONE, latent]
   - Prepare static footprint/range attrs earlier.
   - Keep Wave descriptor materialization in the Wave pass pipeline.
   - Status: `tritonwaveamd-plan-buffer-descriptors` attaches the range; the
     bridge consumes it. The static-affine buffer-ops path is unexercised by the
     current test kernels.

8. Delete compatibility analysis from the bridge. [PARTIAL]
   - Remove TTGIR type-string parsing except structural assertions.
   - Remove schedule, layout, LDS, ownership, and epilogue analysis.
   - Fail fast when preparation did not run.
   - Status: dot-path type-string parsing (opIdx regex, MFMA/convert-layout kind
     string matching) and the schedule/LDS analysis are gone. Layout discovery,
     element-type string parsing, and epilogue folding remain in the bridge.

## Open Questions

- ~~Which structural attribute APIs are available through current Python MLIR
  bindings, and which need C API additions?~~ ANSWERED: `op.get_int_attr` /
  `op.get_str_attr` (return None when absent) cover the prepared-attr reads for
  the passes landed so far; no C API additions were needed.
- How much of Triton AMD intrinsic selection can be factored without depending
  on TritonGPU layout attr construction? Still open -- this is the main unknown
  blocking phase 3 (`add_convert_to_ttgpuir` replacement).
- What is the first MFMA target shape to support once WMMA preparation is
  stable?

## Acceptance Criteria

Status as of 2026-06-09 in brackets.

- [PARTIAL] `lowering.py` contains no GEMM schedule, layout discovery, LDS slot
  selection, buffer range inference, or epilogue preservation analysis. (Schedule,
  LDS slot, and buffer range are gone; layout discovery and epilogue folding
  remain.)
- [PARTIAL] Wave TTIR and TTGIR preparation, including TTIR-to-TTGIR conversion,
  is implemented as Triton-tree C++ passes. (Three TTGIR preparation passes exist;
  TTIR preparation and TTIR-to-TTGIR conversion do not.)
- [NO] Python compiler code is limited to the mechanical TTGIR-to-Wave bridge.
  (`lowering.py` still constructs layouts and folds epilogues.)
- [YES] The Wave pipeline does not call `add_convert_to_ttgpuir` (nor
  `amd.accelerate_matmul`). Both are replaced by Wave-owned passes, though those
  passes still emit TritonGPU layouts rather than Wave-native contracts.
- [PARTIAL] All memory and fragment layout math appears as prepared symbolic
  expressions and lowers to `wave.index_expr`. (Layout math is symbolic
  `wave.index_expr` already, but constructed in the Python bridge, not prepared.)
- [YES] Canonical Triton GEMM examples compile through Wave AMD. f16 GEMM runs
  e2e on gfx1100 with `num_warps` accepted; other HIP autotune knobs
  (`matrix_instr_nonkdim`, `kpack`, `num_stages`) are accepted but not fully
  exercised.
- [PARTIAL] Unsupported layouts and architectures fail in preparation passes with
  stable diagnostics. (MFMA fails in `tritonwaveamd-legalize-dots`; other
  unsupported shapes still fail later in the bridge.)
- [NO] Bridge tests prove lowering is mechanical by feeding already-prepared IR
  and checking Wave dialect output. (Tests feed TTIR and run the full pipeline.)
