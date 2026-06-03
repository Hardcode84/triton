# Wave AMDGPU Backend Implementation Plan

## Scope

- Implement in this Triton branch.
- Include Wave as a submodule for now.
- Backend name: `wave_amd`.
- Target: `GPUTarget("wave_amd", arch, warp_size)`.
- Input boundary: TTIR after minimal cleanup.
- Output boundary: HSACO.
- Runtime: reuse HIP loading/launch path initially.
- Primary lowering: `TTIR -> Wave/WaveAMD MLIR -> AMDGCN -> HSACO`.
- Do not use AMD `make_ttgir()` as the main path.

## Repo Layout

- Implemented:
  - `third_party/wave_amd/`
  - `third_party/wave_amd/backend/name.conf`
  - `third_party/wave_amd/backend/compiler.py`
  - `third_party/wave_amd/backend/emission.py`
  - `third_party/wave_amd/backend/driver.py`
  - `third_party/wave_amd/CMakeLists.txt`
  - `third_party/wave_amd/python/triton_wave_amd.cc`
  - `third_party/wave_amd/wave/`
  - `python/test/unit/runtime/test_wave_amd_backend.py`

- Remaining:
  - `third_party/wave_amd/test/`
  - `backend/lib/` if packaged runtime artifacts are needed

- Implemented Triton setup changes:
  - added `wave_amd` to `BackendInstaller.copy([...])`
  - installs `triton.backends.wave_amd`

- Remaining Triton setup changes:
  - include backend package data

- Implemented Triton CMake changes:
  - builds `third_party/wave_amd` through `TRITON_CODEGEN_BACKENDS`
  - exposes empty native `init_triton_wave_amd` stub
  - builds and packages `wave-translate` under `triton.backends.wave_amd/bin`

- Remaining Triton CMake changes:
  - expose native glue to Python if using in-process emission

## Backend Skeleton

- Implemented `compiler.py`
  - Define `WaveAMDOptions`.
  - Define `WaveAMDBackend(BaseBackend)`.
  - Set `binary_ext = "amdgcn"` for the M2 compile-to-assembly slice.
  - Implement `supports_target(target)`.
  - Implement `parse_options(opts)`.
  - Implement `hash()`.
  - Implement `pack_metadata(metadata)`.
  - Implement `get_codegen_implementation(options)`.
  - Implement `get_module_map()`.
  - Implement `load_dialects(ctx)`.
  - Implement `add_stages(stages, options, language)`.
  - Implement `wave` and `amdgcn` stages.

- Implemented `driver.py`
  - Reuse HIP utility loading where possible.
  - Subclass `HIPDriver`.
  - Return `GPUTarget("wave_amd", arch, warp_size)`.
  - Reuse `HIPLauncher` initially.
  - Keep argument ABI identical to HIP backend.
  - Driver is inactive unless `TRITON_WAVE_AMD_ENABLE=1` and `TRITON_DEFAULT_BACKEND=wave_amd`.

## Compiler Stages

- Implemented stage order:
  - `ttir`
  - `wave`
  - `amdgcn`

- Future stage:
  - `hsaco`

- `make_ttir(mod, metadata, options)`
  - Run safe TTIR cleanup only.
  - Allow:
    - inliner
    - canonicalizer
    - CSE
    - LICM if it preserves address expressions
    - loop unroll only when needed
  - Do not run `convert_to_ttgpuir`.

- `make_wave(mod, metadata, options)`
  - Convert TTIR module to Wave MLIR.
  - Stamp:
    - `gpu.container_module`
    - `waveamdmachine.target`
    - `wave.kernel`
    - `gpu.kernel` if using `gpu.module`
  - Emit Wave/WaveAMD ops.
  - Emit symbolic `wave.index_expr`.
  - Emit memory tokens.
  - Set preliminary metadata.

- `make_amdgcn(wave_module, metadata, options)`
  - Implemented:
    - invoke build-packaged `wave-translate --wave-to-amdgpu-asm`
    - pass Wave MLIR text only as the external compiler artifact
  - Target:
    - call `translateWaveToAMDGPU()` in-process
  - Preserve metadata from earlier stages.

- `make_hsaco(amdgcn, metadata, options)`
  - Initial:
    - assemble with ROCm/LLVM tools or Wave helper path
    - link HSACO
  - Target:
    - call `assembleWaveAMDGPUKernels()` or equivalent in-process path
  - Return bytes.

## TTIR To Wave Bridge

- Implement as native MLIR pass if possible.
- Python bridge is acceptable for first compile-only spike.

- Type mapping:
  - scalar integer/float -> MLIR scalar
  - pointer -> `!wave.ptr`
  - block tensor -> `!wave.simd<T, W>` or Wave tuple/fragment
  - boolean block tensor -> `!wave.mask<W>`
  - matmul fragments -> WaveAMD fragment types

- Operation mapping:
  - `tt.get_program_id` -> `wave.workgroup_id`
  - lane identity -> `wave.workitem_id` / `wave.lane_id`
  - `tt.addptr` -> `wave.ptr_add`
  - pointer offset expressions -> `wave.index_expr`
  - `tt.load` -> `wave.load`
  - `tt.store` -> `wave.store`
  - `tt.atomic_rmw` / `tt.atomic_cas` -> initially unsupported or Wave atomic extension
  - `tt.dot` -> `waveamd.mma`
  - `tt.splat` -> `wave.splat`
  - `tt.where` / masks -> `wave.where`
  - `tt.make_range` -> symbolic lane/tile expression
  - `tt.expand_dims` / `tt.broadcast` / `tt.reshape` -> symbolic shape view when possible
  - `tt.trans` / `tt.permute` -> Wave register/fragment mapping
  - Triton barrier -> `wave.barrier`

## Symbolic Addressing

- Preserve address expressions from TTIR.
- Avoid lowering offsets to opaque integer arithmetic.
- Use Wave symbol store / `ixsimpl` expression attributes.
- Required symbols:
  - `pid_x`
  - `pid_y`
  - `pid_z`
  - `lane`
  - `wave_id`
  - loop induction variables
  - static tile parameters

- Emit `wave.index_expr` for:
  - lane modulo
  - lane division
  - wave id decomposition
  - tile coordinates
  - strides
  - vectorized contiguous offsets

## Memory Tokens

- Add bridge-side `TokenState`.
- Initial token state:
  - one token per memory space per block
  - `global`
  - `shared`
  - optional `private`

- At function entry:
  - create `wave.token` per tracked memory space.

- For `tt.store`:
  - consume current token for pointer address space.
  - update that token with returned `wave.store` token.

- For `tt.load`:
  - ordinary non-volatile load:
    - use no token unless needed for explicit ordering.
  - volatile load:
    - consume and update token.

- For atomics:
  - consume and update token.
  - respect `sem` / `scope` when Wave atomic support exists.
  - reject initially if unsupported.

- For DMA/global-to-LDS:
  - consume joined global/shared token.
  - update both spaces from returned token.

- For barrier:
  - join outstanding relevant tokens.
  - emit `wave.barrier`.
  - update shared/global token as needed from barrier result.

- For control flow:
  - branch exit tokens merge with `wave.join`.
  - loop-carried token state becomes loop-carried SSA.
  - unsupported cases fail with diagnostic.

## Metadata

- Populate:
  - `name`
  - `num_warps`
  - `num_ctas`
  - `shared`
  - `warp_size`
  - `global_scratch_size`
  - `global_scratch_align`
  - `profile_scratch_size`
  - `profile_scratch_align`
  - `tensordesc_meta`

- Source of values:
  - `name`: Wave kernel symbol
  - `shared`: `wave.lds_size` or Wave resource metadata
  - `num_warps`: backend options / Wave target waves
  - scratch fields: zero until supported
  - tensor descriptor fields: unsupported until ABI is proven

## ABI Checks

- Match HIP launcher flattened argument ABI.
- Check scalar sizes.
- Check pointer address spaces.
- Check bool handling.
- Check tuple flattening.
- Check tensor descriptors separately.
- Reject kernels with unsupported ABI features.

## Tests

- Implemented tests:
  - backend skeleton options/stages
  - one concrete driver class
  - explicit driver activation gate

- Compile-only tests:
  - TTIR cleanup stops before TTGIR
  - Wave MLIR dump contains `wave.kernel`
  - Wave MLIR dump contains `wave.index_expr`
  - Wave MLIR dump contains `wave.mem.token`
  - AMDGCN emitted with a packaged fake `wave-translate`
  - missing packaged `wave-translate` diagnostic
  - HSACO emitted

- Runtime tests:
  - masked load/store
  - strided copy
  - simple elementwise add
  - barrier + LDS smoke test
  - tiled matmul once fragments work

- Negative tests:
  - unsupported atomic
  - unsupported tensor descriptor
  - unsupported control-flow token merge
  - unsupported block tensor shape
  - ABI mismatch

## Milestones

- M0: done. Backend skeleton registered as `wave_amd`; selection is gated behind explicit env vars.
- M1: compile a supported TTIR op subset from live module IR to builder-generated Wave MLIR; no TTIR assembly parsing or kernel-shape matching.
- M2: done. Emit AMDGCN assembly from Wave MLIR through `wave-translate`.
- M3: emit HSACO and load through HIP runtime.
- M4: run masked load/store kernel end-to-end.
- M5: add token threading for stores, volatile loads, barriers, and simple branches.
- M6: add symbolic strided copy.
- M7: add WaveAMD matmul path.
- M8: replace submodule tool subprocesses with in-process Wave APIs.

## Open Work Items

- Choose TTIR cleanup pass list.
- Define block tensor to Wave SIMD layout contract.
- Define token merge representation for structured control flow.
- Decide first supported atomic subset.
- Decide first supported tensor descriptor subset.
- Decide exact submodule path and build artifact discovery.
- Decide whether TTIR-to-Wave bridge lands in Triton backend glue or Wave submodule.
