import { describe, expect, it } from 'vitest'
import {
  GB,
  GBps,
  GiB,
  KiB,
  TBps,
  TFLOPS,
  achievedBandwidth,
  addBytes,
  bitsToBytes,
  bytes,
  cluster,
  computeTime,
  flops,
  intensity,
  microseconds,
  seconds,
  share,
  toMilliseconds,
  total,
  transferTime,
} from './units'

describe('construction', () => {
  it('refuses values that cannot describe a physical quantity', () => {
    expect(() => bytes(Number.NaN)).toThrow(RangeError)
    expect(() => bytes(Number.POSITIVE_INFINITY)).toThrow(RangeError)
    expect(() => bytes(-1)).toThrow(RangeError)
    expect(() => flops(-1e12)).toThrow(RangeError)
    expect(() => seconds(-0.001)).toThrow(RangeError)
  })

  it('accepts zero, which is a real quantity and not a missing one', () => {
    expect(bytes(0)).toBe(0)
    expect(flops(0)).toBe(0)
  })
})

describe('scope', () => {
  // The regression this whole module exists for: an 8x error from quoting a
  // cluster-wide KV-cache read as if it were the traffic on one GPU.
  it('divides a cluster total into per-rank shares', () => {
    const clusterKvRead = cluster(GB(21.47))
    // Compared in GB: toBeCloseTo takes an absolute tolerance, which says
    // nothing useful about two numbers of order 1e9.
    expect(share(clusterKvRead, 8) / 1e9).toBeCloseTo(2.684, 2)
  })

  it('round-trips share and total', () => {
    const each = share(cluster(GB(160)), 8)
    expect(total(each, 8)).toBeCloseTo(GB(160), 0)
  })

  it('rejects a world size that is not a positive integer', () => {
    expect(() => share(cluster(GB(1)), 0)).toThrow(RangeError)
    expect(() => share(cluster(GB(1)), 2.5)).toThrow(RangeError)
    expect(() => total(share(cluster(GB(1)), 1), -8)).toThrow(RangeError)
  })
})

describe('time', () => {
  it('derives transfer time from bytes and bandwidth', () => {
    // 20.2 GB of decode traffic across HBM at 2.68 TB/s effective.
    const t = transferTime(GB(20.184), TBps(2.68))
    expect(toMilliseconds(t)).toBeCloseTo(7.53, 1)
  })

  it('treats moving nothing as taking no time, whatever the rate', () => {
    expect(transferTime(bytes(0), GBps(0))).toBe(0)
    expect(computeTime(flops(0), TFLOPS(0))).toBe(0)
  })

  it('refuses to move bytes at a rate of zero', () => {
    expect(() => transferTime(GB(1), GBps(0))).toThrow(RangeError)
    expect(() => computeTime(flops(1e12), TFLOPS(0))).toThrow(RangeError)
  })
})

describe('intensity', () => {
  it('reports the H100 BF16 ridge point', () => {
    // 989.5 TFLOP/s dense over 3.35 TB/s is the crossover intensity at which a
    // BF16 kernel stops being memory-bound on an H100 SXM5.
    const ridge = intensity(flops(989.5e12), bytes(3.35e12))
    expect(ridge).toBeCloseTo(295.4, 1)
  })

  it('calls a kernel that touches no memory infinitely intense', () => {
    expect(intensity(flops(1e12), bytes(0))).toBe(Number.POSITIVE_INFINITY)
  })

  it('reports no intensity at all rather than inventing one', () => {
    // 0/0 has no answer. NaN plots as a hole; 0 or Infinity would plot as a lie.
    expect(intensity(flops(0), bytes(0))).toBeNaN()
  })
})

describe('scale helpers', () => {
  it('keeps decimal and binary units distinct', () => {
    // NVIDIA's "80 GB" H100 is 80e9 bytes, which is only ~74.5 GiB. Treating the
    // two as interchangeable is a 7.4% capacity error.
    expect(GB(80)).toBe(80e9)
    expect(GiB(80)).toBe(85_899_345_920)
    expect(GB(80) / GiB(1)).toBeCloseTo(74.5, 1)
  })

  it('converts interconnect bit rates to byte rates', () => {
    // 8x ConnectX-7 at 400 Gb/s each is 3.2 Tb/s, which is 400 GB/s.
    expect(bitsToBytes(3.2e12)).toBe(400e9)
  })

  it('sizes a batch-1 tensor-parallel all-reduce payload', () => {
    // One token, hidden size 8192, bf16: the message that makes TP=8 decode
    // latency-bound rather than bandwidth-bound.
    expect(bytes(1 * 8192 * 2)).toBe(KiB(16))
  })
})

describe('achieved rates', () => {
  it('derives memory bandwidth utilisation from bytes and elapsed time', () => {
    const achieved = achievedBandwidth(GB(20.184), microseconds(7060))
    const mbu = achieved / TBps(3.35)
    expect(mbu).toBeCloseTo(0.853, 2)
  })
})

describe('sums', () => {
  it('adds per-component byte counts', () => {
    // Per-GPU decode traffic: 17.5 GB of weights plus 2.68 GB of KV.
    expect(addBytes(GB(17.5), GB(2.684))).toBeCloseTo(GB(20.184), 0)
  })

  it('sums nothing to zero', () => {
    expect(addBytes()).toBe(0)
  })
})
