/**
 * The accelerator catalog.
 *
 * Every figure here is a vendor specification with a citation, or it does not
 * belong in this file. Fitted constants — achievable fractions of peak,
 * collective latency floors — live in `efficiency.ts` and `collectives.ts`
 * respectively, and are labelled priors there. Keeping the two apart is the
 * whole point: a user who wants to argue with this model should be able to see
 * instantly which numbers are NVIDIA's and which are ours.
 */

import {
  type Bytes,
  type BytesPerSecond,
  type FlopsPerSecond,
  GBps,
  GiB,
  MiB,
  TBps,
  TFLOPS,
  bitsToBytes,
} from './units'

/**
 * Precisions a tensor core can compute in.
 *
 * `tf32` is a compute mode, not a storage format: inputs are fp32 in memory and
 * rounded to 19 bits inside the tensor core. It appears here because it has its
 * own peak rate, and is absent from `Dtype` because nothing is stored as tf32.
 */
export type TensorPrecision = 'fp32' | 'tf32' | 'bf16' | 'fp16' | 'fp8' | 'int8'

/** Storage formats, for weights, activations and KV cache. */
export type Dtype = 'fp32' | 'bf16' | 'fp16' | 'fp8' | 'int8' | 'int4'

/** Bytes per element. int4 is half a byte; callers work in bytes, not elements. */
export const DTYPE_BYTES: Record<Dtype, number> = {
  fp32: 4,
  bf16: 2,
  fp16: 2,
  fp8: 1,
  int8: 1,
  int4: 0.5,
}

/** The tensor-core rate a stored dtype computes at. */
export const DTYPE_COMPUTES_AS: Record<Dtype, TensorPrecision> = {
  fp32: 'fp32',
  bf16: 'bf16',
  fp16: 'fp16',
  fp8: 'fp8',
  int8: 'int8',
  // Weight-only 4-bit schemes dequantize to 16-bit before the tensor core sees
  // them, so W4A16 computes at bf16 rate while moving a quarter of the bytes.
  // That asymmetry is precisely what shifts decode off the memory roof.
  int4: 'bf16',
}

export interface Interconnect {
  readonly name: string
  /** Usable bandwidth in ONE direction, per GPU. The figure roofline math wants. */
  readonly perDirection: BytesPerSecond
  /** Vendor headline, the sum of both directions. Present only to be recognised. */
  readonly bidirectional: BytesPerSecond
  readonly links: number
  readonly perLinkPerDirection: BytesPerSecond
}

export interface Accelerator {
  readonly id: string
  readonly name: string
  readonly vendor: string

  /**
   * Dense peak rates. NVIDIA publishes only the WITH-SPARSITY figure for
   * tf32/bf16/fp16/fp8/int8, footnoted with an asterisk; dense is half of it.
   * 2:4 structured sparsity needs explicitly pruned weights and is not used in
   * production LLM work, so dense is what this catalog stores and what every
   * ceiling is drawn from.
   */
  readonly peak: Readonly<Record<TensorPrecision, FlopsPerSecond>>

  /** Physical HBM. Binary units: the "80 GB" H100 is 80 GiB. */
  readonly memoryCapacity: Bytes
  /** What the driver actually exposes, after ECC and context reserve. */
  readonly memoryAddressable: Bytes
  readonly memoryBandwidth: BytesPerSecond
  readonly memoryType: string

  readonly sms: number
  readonly l2Cache: Bytes
  readonly interconnect: Interconnect
  readonly pciePerDirection: BytesPerSecond
  readonly tdpWatts: number
  readonly sources: readonly string[]
}

/** Sparse peak, for the one purpose of showing users what the asterisk hides. */
export const sparsePeak = (dense: FlopsPerSecond): FlopsPerSecond =>
  (dense * 2) as FlopsPerSecond

const NVIDIA_SOURCES = [
  'https://www.nvidia.com/en-us/data-center/h100/',
  'NVIDIA H100 Tensor Core GPU Architecture whitepaper v1.02, Table 3',
] as const

/**
 * NVLink 4, stated exactly as the Hopper whitepaper does:
 *
 *   "fourth-generation NVLink uses only two high-speed differential pairs in
 *    each direction to form a single link, also delivering 25 GB/sec effective
 *    bandwidth in each direction. H100 includes 18 fourth-generation NVLink
 *    links to provide 900 GB/sec total bandwidth"
 *
 * So 18 x 25 = 450 GB/s each way, and the headline 900 GB/s is the sum of both
 * directions. Using 900 where 450 belongs is the classic 2x error in this field.
 */
const NVLINK4: Interconnect = {
  name: 'NVLink 4',
  perDirection: GBps(450),
  bidirectional: GBps(900),
  links: 18,
  perLinkPerDirection: GBps(25),
}

/**
 * The PCIe and NVL parts do not carry the SXM5's full 18-link NVLink. Both are
 * rated 600 GB/s bidirectional over a bridge — 300 GB/s each way, 12 links.
 * Giving them the SXM figure overstates intra-node bandwidth by 1.5x.
 */
const NVLINK4_BRIDGE: Interconnect = {
  name: 'NVLink 4 (bridge)',
  perDirection: GBps(300),
  bidirectional: GBps(600),
  links: 12,
  perLinkPerDirection: GBps(25),
}

/** PCIe Gen5 x16: "128 GB/sec total bandwidth (64 GB/sec in each direction)". */
const PCIE5_X16 = GBps(64)

export const H100_SXM5: Accelerator = {
  id: 'h100-sxm5',
  name: 'H100 SXM5 80GB',
  vendor: 'NVIDIA',
  peak: {
    fp32: TFLOPS(67),
    tf32: TFLOPS(494.5),
    bf16: TFLOPS(989.5),
    fp16: TFLOPS(989.5),
    fp8: TFLOPS(1979),
    int8: TFLOPS(1979),
  },
  // Five active HBM3 stacks of 16 GiB. nvidia-smi reports 81,559 MiB, which is
  // 80 GiB less 361 MiB of ECC and context reserve.
  memoryCapacity: GiB(80),
  memoryAddressable: MiB(81_559),
  memoryBandwidth: TBps(3.35),
  memoryType: 'HBM3',
  sms: 132,
  l2Cache: MiB(50),
  interconnect: NVLINK4,
  pciePerDirection: PCIE5_X16,
  tdpWatts: 700,
  sources: NVIDIA_SOURCES,
}

export const H100_PCIE: Accelerator = {
  id: 'h100-pcie',
  name: 'H100 PCIe 80GB',
  vendor: 'NVIDIA',
  peak: {
    fp32: TFLOPS(51),
    tf32: TFLOPS(378),
    bf16: TFLOPS(756.5),
    fp16: TFLOPS(756.5),
    fp8: TFLOPS(1513),
    int8: TFLOPS(1513),
  },
  memoryCapacity: GiB(80),
  memoryAddressable: MiB(81_559),
  memoryBandwidth: TBps(2.0),
  memoryType: 'HBM2e',
  sms: 114,
  l2Cache: MiB(50),
  interconnect: NVLINK4_BRIDGE,
  pciePerDirection: PCIE5_X16,
  tdpWatts: 350,
  sources: NVIDIA_SOURCES,
}

export const H100_NVL: Accelerator = {
  id: 'h100-nvl',
  name: 'H100 NVL 94GB',
  vendor: 'NVIDIA',
  peak: {
    fp32: TFLOPS(60),
    tf32: TFLOPS(417.5),
    bf16: TFLOPS(835.5),
    fp16: TFLOPS(835.5),
    fp8: TFLOPS(1670.5),
    int8: TFLOPS(1670.5),
  },
  // Six stacks with more of each die enabled than SXM5; the vendor figure is
  // 94 GB and the binary reading (94 GiB) is not corroborated, so the decimal
  // reading is used and flagged.
  memoryCapacity: GiB(94),
  memoryAddressable: MiB(95_830),
  memoryBandwidth: TBps(3.9),
  memoryType: 'HBM3',
  sms: 132,
  l2Cache: MiB(50),
  interconnect: NVLINK4_BRIDGE,
  pciePerDirection: PCIE5_X16,
  tdpWatts: 400,
  sources: [
    ...NVIDIA_SOURCES,
    'H100 NVL Product Brief PB-11773-001',
  ],
}

export const ACCELERATORS: readonly Accelerator[] = [H100_SXM5, H100_PCIE, H100_NVL]

/** How the GPUs in a system are wired to each other. */
export type Fabric = 'single' | 'nvswitch' | 'nvlink-bridge' | 'pcie'

export interface Topology {
  readonly id: string
  readonly name: string
  readonly accelerator: Accelerator
  readonly gpus: number
  readonly fabric: Fabric
  /** NVSwitch chips, where the fabric has any. */
  readonly switches?: number
  /** Aggregate GPU-to-GPU, both directions summed. Vendor headline. */
  readonly aggregateBidirectional?: BytesPerSecond
  /** Bisection, both directions summed. */
  readonly bisectionBidirectional?: BytesPerSecond
  /** Per-node network out of the box, one direction. */
  readonly nodeNetworkPerDirection?: BytesPerSecond
  readonly sources: readonly string[]
}

export const SINGLE_H100: Topology = {
  id: 'h100-x1',
  name: '1x H100 SXM5',
  accelerator: H100_SXM5,
  gpus: 1,
  fabric: 'single',
  sources: NVIDIA_SOURCES,
}

/**
 * DGX/HGX H100, eight GPUs on one baseboard.
 *
 * The whitepaper's own numbers are self-consistent at 450 GB/s per GPU per
 * direction: 8 x 450 x 2 = 7.2 TB/s aggregate, and 4 x 450 x 2 = 3.6 TB/s
 * bisection. NVIDIA also quotes "13.6 Tbit/s" per NVSwitch, which reconciles
 * with neither convention cleanly, so nothing here is built on it.
 *
 * The "32 PFLOPS" system headline is SPARSE FP8. Dense FP8 is 15.8 PF and dense
 * BF16 is 7.9 PF; a tool repeating 32 PF is out by 2x or 4x.
 */
export const DGX_H100: Topology = {
  id: 'h100-x8',
  name: '8x H100 SXM5 (NVSwitch)',
  accelerator: H100_SXM5,
  gpus: 8,
  fabric: 'nvswitch',
  switches: 4,
  aggregateBidirectional: TBps(7.2),
  bisectionBidirectional: TBps(3.6),
  // 8 x ConnectX-7 at 400 Gb/s = 3.2 Tb/s = 400 GB/s each direction.
  nodeNetworkPerDirection: bitsToBytes(3.2e12),
  sources: [
    ...NVIDIA_SOURCES,
    'NVIDIA DGX H100 User Guide',
    'Introducing NVIDIA HGX H100 (NVIDIA developer blog)',
  ],
}

export const TOPOLOGIES: readonly Topology[] = [SINGLE_H100, DGX_H100]

/** Aggregate dense peak across the topology, at a given precision. */
export function clusterPeak(topology: Topology, precision: TensorPrecision): FlopsPerSecond {
  return (topology.accelerator.peak[precision] * topology.gpus) as FlopsPerSecond
}

/** Aggregate HBM capacity across the topology, as addressable bytes. */
export function clusterMemory(topology: Topology): Bytes {
  return (topology.accelerator.memoryAddressable * topology.gpus) as Bytes
}

/**
 * The machine balance: FLOPs per byte at which a kernel stops being memory-bound.
 *
 * This single number explains more about LLM performance than any other. For an
 * H100 SXM5 in bf16 it is ~295, which is why a decode step running at an
 * intensity near 1 is memory-bound by more than two orders of magnitude.
 */
export function ridgePoint(accelerator: Accelerator, precision: TensorPrecision): number {
  return accelerator.peak[precision] / accelerator.memoryBandwidth
}
