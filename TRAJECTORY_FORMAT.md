# MUrB trajectory format

`.murbtraj` version 1 is a little-endian, field-by-field binary format. Numeric
fields are never written as native C++ structs or native-width integer types.
The current scalar code supports IEEE-754 binary32 values only.

## Header and static-body section

| Offset | Size | Field |
|---:|---:|---|
| 0 | 8 | Magic bytes `MURBTRJ\0` |
| 8 | 4 | Format version, unsigned 32-bit; currently `1` |
| 12 | 4 | Endianness policy marker, unsigned 32-bit; `0x01020304` |
| 16 | 4 | Scalar code, unsigned 32-bit; `1` means fp32 |
| 20 | 4 | Static-body flags, unsigned 32-bit; `1` means radii are present |
| 24 | 8 | Total header/static-section byte count, unsigned 64-bit |
| 32 | 8 | Body count, unsigned 64-bit |
| 40 | 8 | Actual frame count, unsigned 64-bit |
| 48 | 8 | Recording stride in timed iterations, unsigned 64-bit |
| 56 | 8 | Simulation timestep, IEEE-754 binary64 |
| 64 | 4 | Backend-name byte count, unsigned 32-bit |
| 68 | 4 | Source-commit byte count, unsigned 32-bit |
| 72 | variable | UTF-8 backend bytes, without a terminator |
| next | variable | UTF-8 source Git commit bytes, without a terminator |
| next | `4*N` | Body radii as `N` fp32 values |

The writer initially stores a zero frame count and replaces it with the actual
count when the file is finalized. A partially written or unfinalized file does
not pass the reader's exact file-size validation.

## Frame layout

Each frame has exactly this layout:

| Size | Field |
|---:|---|
| 8 | Timed iteration number, unsigned 64-bit |
| `4*N` | `qx[0..N-1]`, fp32 |
| `4*N` | `qy[0..N-1]`, fp32 |
| `4*N` | `qz[0..N-1]`, fp32 |
| `4*N` | `vx[0..N-1]`, fp32 |
| `4*N` | `vy[0..N-1]`, fp32 |
| `4*N` | `vz[0..N-1]`, fp32 |

The frame size is `8 + 24*N` bytes. Frames are recorded after a timed
iteration completes and, for CUDA backends, after its synchronization. Warm-up
iterations are not recorded and do not affect the stored frame iteration
numbers.

Readers must reject unknown versions, scalar codes, endianness markers or
flags. They must also require the physical file size to equal the declared
header size plus `frame_count * frame_size` before publishing any frame.

The historical unversioned `simulation_data.bin` layout is not part of this
format and is intentionally not auto-detected.
