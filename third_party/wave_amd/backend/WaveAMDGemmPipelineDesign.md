# Wave AMD GEMM Pipeline Design

## Status

Design proposal for the Wave AMD Triton backend GEMM path.

Current code proves the pieces can work: Triton TTIR/TTGIR reaches the Wave
dialect, dot operands can be staged through LDS, and WMMA fragments lower to
Wave AMD machine code. The next step is to move policy and analysis out of the
TTGIR-to-Wave bridge and into explicit preparation passes.

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

1. Freeze the bridge contract.
   - Add bridge tests that require prepared attrs for GEMM lowering.
   - Keep current behavior only behind temporary compatibility helpers.

2. Move architecture policy out of `lowering.py`.
   - Add a Wave TTGIR legalization step.
   - Reuse Triton AMD target feature and intrinsic legality logic where possible.
   - Fix option plumbing so `matrix_instr_nonkdim` has one default.

3. Replace `add_convert_to_ttgpuir`.
   - Add a Triton-tree C++ TTIR-to-TTGIR conversion for Wave.
   - Emit Wave-compatible layout contracts directly.
   - Remove dependence on TritonGPU layout approximations for the Wave path.

4. Move layout expression construction out of `lowering.py`.
   - Introduce symbolic layout/index metadata in TTIR/TTGIR preparation.
   - Lower those expressions to `wave.index_expr` in the bridge.
   - Remove bridge-side blocked layout reconstruction.

5. Move dot planning out of `lowering.py`.
   - Emit explicit dot plans with operand roles, fragment geometry, and
     accumulator layout.
   - Bridge maps plans to fragment pack/unpack and `waveamd.mma`.

6. Move LDS and software pipeline planning out of `lowering.py`.
   - Turn `WaveGemmSchedule` into preparation output.
   - Attach stage, LDS slot, token, and resource attrs before bridge lowering.
   - Bridge emits only the prepared Wave ops.

7. Move buffer descriptor planning out of `lowering.py`.
   - Prepare static footprint/range attrs earlier.
   - Keep Wave descriptor materialization in the Wave pass pipeline.

8. Delete compatibility analysis from the bridge.
   - Remove TTGIR type-string parsing except structural assertions.
   - Remove schedule, layout, LDS, ownership, and epilogue analysis.
   - Fail fast when preparation did not run.

## Open Questions

- Which structural attribute APIs are available through current Python MLIR
  bindings, and which need C API additions?
- How much of Triton AMD intrinsic selection can be factored without depending
  on TritonGPU layout attr construction?
- What is the first MFMA target shape to support once WMMA preparation is
  stable?

## Acceptance Criteria

- `lowering.py` contains no GEMM schedule, layout discovery, LDS slot selection,
  buffer range inference, or epilogue preservation analysis.
- Wave TTIR and TTGIR preparation, including TTIR-to-TTGIR conversion, is
  implemented as Triton-tree C++ passes.
- Python compiler code is limited to the mechanical TTGIR-to-Wave bridge.
- The Wave pipeline does not call `add_convert_to_ttgpuir`.
- All memory and fragment layout math appears as prepared symbolic expressions
  and lowers to `wave.index_expr`.
- Canonical Triton GEMM examples compile through Wave AMD with HIP autotune
  knobs accepted.
- Unsupported layouts and architectures fail in preparation passes with stable
  diagnostics.
- Bridge tests prove lowering is mechanical by feeding already-prepared IR and
  checking Wave dialect output.
