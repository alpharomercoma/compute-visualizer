import { describe, expect, it } from 'vitest'
import { DGX_H100, H100_SXM5, ridgePoint } from './hardware'
import { H100_EFFICIENCY, H100_OVERHEAD } from './efficiency'
import { type HuggingFaceConfig, fromHuggingFaceConfig } from './modelspec'
import { analyseInference, computeBoundBatch } from './inference'
import { chooseCollective, busbwToAlgbw, defaultProfiles, latencyBandwidthKnee } from './collectives'
import { attentionPairs } from './kernels'
import { GBps, KiB, MiB, toMilliseconds } from './units'

/**
 * The golden scenarios from docs/MODEL.md section 14.
 *
 * These are the tests that decide whether the model is worth anything. Scenario
 * 1 was derived three independent ways during design review — by hand, by codex,
 * and by agy — and the disagreement between them (an 8x error in one reviewer's
 * KV figure) is the reason the numbers below are pinned rather than trusted.
 *
 * Every bound is a RANGE. Implementation choices legitimately move these by tens
 * of percent, and a test asserting a point value would be asserting a fiction.
 */

const LLAMA_3_70B: HuggingFaceConfig = {
  model_type: 'llama',
  hidden_size: 8192,
  num_hidden_layers: 80,
  num_attention_heads: 64,
  num_key_value_heads: 8,
  intermediate_size: 28672,
  vocab_size: 128256,
  max_position_embeddings: 8192,
  tie_word_embeddings: false,
  torch_dtype: 'bfloat16',
}

const llama70b = fromHuggingFaceConfig(LLAMA_3_70B, { id: 'llama-3-70b' })

const baseConfig = {
  spec: llama70b,
  topology: DGX_H100,
  parallelism: { tensor: 8, pipeline: 1, data: 1 },
  weightDtype: 'bf16',
  kvDtype: 'bf16',
  activationDtype: 'bf16',
  efficiency: H100_EFFICIENCY,
  overhead: H100_OVERHEAD,
  cudaGraph: true,
  flashAttention: true,
} as const

describe('scenario 1: Llama-3-70B BF16, 8xH100 TP=8, batch 32, 2048 context', () => {
  const result = analyseInference({
    ...baseConfig,
    batch: 32,
    promptLength: 2048,
    generateLength: 1,
  })

  it('fits, and reports where the memory went', () => {
    expect(result.feasible).toBe(true)
    const { weights, kvCache } = result.capacity.perGpu

    // 70.6B parameters at 2 bytes over 8 ranks.
    expect(weights / 1e9).toBeCloseTo(17.6, 0)
    // The figure a reviewer got 8x wrong: 2 * 32 * 2048 * 80 * 128 * 2 bytes.
    expect(kvCache / 1e9).toBeCloseTo(2.68, 1)
  })

  it('produces a decode step in the 6-11 ms band', () => {
    const ms = toMilliseconds(result.timePerOutputToken!)
    expect(ms).toBeGreaterThan(6)
    expect(ms).toBeLessThan(11)
  })

  it('is memory-bandwidth-bound, not compute-bound', () => {
    expect(result.decode!.bottleneck).toBe('memory-bandwidth')
  })

  it('reaches high MBU and low MFU, which is the correct answer', () => {
    // Decode streams weights and KV while doing almost no arithmetic. A high MFU
    // here would mean the model is wrong, not that the hardware is fast.
    //
    // MBU does not reach the ~87% streaming ceiling because roughly an eighth of
    // the step is collectives, during which no HBM traffic is being credited.
    expect(result.decodeMbu!).toBeGreaterThan(0.65)
    expect(result.decodeMbu!).toBeLessThan(0.85)
    // 4.51 TFLOP against 7.912 PF/s over ~8.5 ms is ~6.7%.
    expect(result.decodeMfu!).toBeGreaterThan(0.04)
    expect(result.decodeMfu!).toBeLessThan(0.1)
  })

  it('serves several thousand tokens per second in aggregate', () => {
    expect(result.aggregateTokensPerSecond!).toBeGreaterThan(2800)
    expect(result.aggregateTokensPerSecond!).toBeLessThan(5500)
  })

  it('sits far below the ridge point in arithmetic intensity', () => {
    const ridge = ridgePoint(H100_SXM5, 'bf16')
    expect(ridge).toBeCloseTo(295, 0)
    expect(result.decode!.arithmeticIntensity).toBeLessThan(ridge / 5)
  })

  it('makes prefill compute-bound, the opposite of decode', () => {
    expect(result.prefill!.bottleneck).toBe('compute')
    expect(result.prefill!.arithmeticIntensity).toBeGreaterThan(result.decode!.arithmeticIntensity)
  })

  it('gives a TTFT of a couple of seconds for 65k prompt tokens', () => {
    const ttft = result.timeToFirstToken!
    expect(ttft).toBeGreaterThan(1.2)
    expect(ttft).toBeLessThan(5)
  })

  it('reaches a plausible prefill MFU', () => {
    // Published end-to-end training MFU on H100 is 38-43%; a compute-bound
    // prefill should land in the same neighbourhood, and certainly not above the
    // large-GEMM efficiency ceiling.
    expect(result.prefillMfu!).toBeGreaterThan(0.25)
    expect(result.prefillMfu!).toBeLessThanOrEqual(H100_EFFICIENCY.gemmLarge.value)
  })
})

describe('scenario 2: batch-1 decode is latency-bound, not bandwidth-bound', () => {
  it('classifies the collective as latency-bound at 16 KiB', () => {
    const profiles = defaultProfiles(8, true)
    const choice = chooseCollective(KiB(16), profiles)
    expect(choice.latencyBound).toBe(true)
  })

  it('puts the knee about two orders of magnitude above the message size', () => {
    const profiles = defaultProfiles(8, true)
    const nvls = profiles.find((p) => p.algorithm === 'nvls')!
    const knee = latencyBandwidthKnee(nvls)

    // alpha 5 us x 274 GB/s is about 1.4 MB.
    expect(knee / MiB(1)).toBeGreaterThan(0.8)
    expect(knee / MiB(1)).toBeLessThan(2.5)
    // A batch-1 TP=8 all-reduce is 16 KiB, far below it.
    expect(knee / KiB(16)).toBeGreaterThan(50)
  })

  it('spends far more on startup than on the wire', () => {
    const profiles = defaultProfiles(8, true)
    const nvls = profiles.find((p) => p.algorithm === 'nvls')!
    const transfer = KiB(16) / nvls.algbw
    expect(nvls.alpha / transfer).toBeGreaterThan(50)
  })
})

describe('scenario 3: the tensor-parallel scaling limit', () => {
  const llama8b = fromHuggingFaceConfig(
    {
      model_type: 'llama',
      hidden_size: 4096,
      num_hidden_layers: 32,
      num_attention_heads: 32,
      num_key_value_heads: 8,
      intermediate_size: 14336,
      vocab_size: 128256,
      max_position_embeddings: 8192,
      tie_word_embeddings: false,
    },
    { id: 'llama-3.1-8b' },
  )

  const sweep = [1, 2, 4, 8].map((tensor) => ({
    tensor,
    result: analyseInference({
      ...baseConfig,
      spec: llama8b,
      parallelism: { tensor, pipeline: 1, data: 1 },
      batch: 1,
      promptLength: 512,
      generateLength: 128,
    }),
  }))

  it('keeps getting faster with more ranks, but by less each time', () => {
    const steps = sweep.map((s) => s.result.timePerOutputToken!)
    for (let i = 1; i < steps.length; i++) {
      expect(steps[i]!).toBeLessThan(steps[i - 1]!)
    }
    const firstSpeedup = steps[0]! / steps[1]!
    const lastSpeedup = steps[2]! / steps[3]!
    // Doubling ranks never quite doubles throughput, and the shortfall grows.
    expect(firstSpeedup).toBeLessThan(2)
    expect(lastSpeedup).toBeLessThan(firstSpeedup)
  })

  it('holds collective latency constant while memory time halves', () => {
    // This is the mechanism behind the diminishing returns: latency is a
    // per-collective floor that does not care how many ranks share the work.
    const latencies = sweep.slice(1).map((s) => s.result.decode!.schedule.breakdown['network-latency'])
    for (const latency of latencies) {
      expect(latency).toBeCloseTo(latencies[0]!, 4)
    }

    const memoryTimes = sweep.map((s) => s.result.decode!.schedule.breakdown.memory)
    expect(memoryTimes[3]! / memoryTimes[0]!).toBeLessThan(0.2)
  })

  it('grows latency as a share of the step, without ever becoming bandwidth-bound', () => {
    const share = (i: number) => {
      const r = sweep[i]!.result
      return r.decode!.schedule.breakdown['network-latency'] / r.timePerOutputToken!
    }
    expect(share(1)).toBeLessThan(share(2))
    expect(share(2)).toBeLessThan(share(3))
    expect(share(3)).toBeGreaterThan(0.2)

    // Nothing here is limited by interconnect bandwidth at any rank count, which
    // is exactly why "communication-bound" as one category is not good enough.
    for (const { result } of sweep) {
      expect(result.decode!.schedule.breakdown['network-bandwidth']).toBe(0)
    }
  })
})

describe('the busbw double-counting trap', () => {
  it('converts a published ring busbw to algbw exactly once', () => {
    // 370 GB/s busbw at 8 ranks is 370 / 1.75 = 211 GB/s algbw.
    const algbw = busbwToAlgbw(GBps(370), 'all-reduce', 8)
    expect(algbw / 1e9).toBeCloseTo(211.4, 0)
  })

  it('leaves NVLS algbw below the NVLink line rate', () => {
    // The reported 480 GB/s NVLS busbw exceeds the 450 GB/s line rate, which is
    // itself proof the ring conversion is wrong for in-network reduction. The
    // algbw the model stores must be physically possible.
    const profiles = defaultProfiles(8, true)
    const nvls = profiles.find((p) => p.algorithm === 'nvls')!
    expect(nvls.algbw).toBeLessThan(H100_SXM5.interconnect.perDirection)
  })

  it('does not double-apply the ring factor', () => {
    const ring = defaultProfiles(8, true).find((p) => p.algorithm === 'ring')!
    // If the factor were applied twice, algbw would be 370/1.75/1.75 = 121.
    expect(ring.algbw / 1e9).toBeGreaterThan(200)
  })
})

describe('the causal-mask discount', () => {
  it('halves a full prefill', () => {
    const s = 2048
    expect(attentionPairs(s, s, true) / (s * s)).toBeCloseTo(0.5, 2)
  })

  it('does not discount decode at all', () => {
    // One new query attends the entire existing context. Applying a blanket /2
    // here understates decode attention by 2x.
    const s = 2048
    expect(attentionPairs(1, s, true)).toBe(s)
  })

  it('discounts a mid-sequence prefill chunk only partially', () => {
    // A 512-token chunk landing at position 1536 attends its prefix in full.
    const pairs = attentionPairs(512, 2048, true)
    expect(pairs).toBeGreaterThan(512 * 1536)
    expect(pairs).toBeLessThan(512 * 2048)
  })
})

describe('the compute-bound batch threshold', () => {
  it('lands near the machine balance for bf16', () => {
    // A weight-stationary GEMM crosses the ridge when batch reaches
    // (peak/bandwidth) * w/2, which for bf16 on H100 is ~295.
    const threshold = computeBoundBatch(DGX_H100, 'bf16', 'bf16')
    expect(threshold).toBeCloseTo(295, -1)
  })

  it('barely moves for fp8, because the ridge point moves too', () => {
    // fp8 halves the bytes but doubles the peak, so the crossover is unchanged.
    const bf16 = computeBoundBatch(DGX_H100, 'bf16', 'bf16')
    const fp8 = computeBoundBatch(DGX_H100, 'fp8', 'fp8')
    expect(fp8).toBeCloseTo(bf16, -1)
  })
})

describe('pipeline parallelism', () => {
  /**
   * The defect this guards: a forward pass is built for the layers ONE stage
   * owns, so its makespan is a stage time. Reporting that as the step time
   * underpredicts latency by roughly the pipeline degree.
   */
  it('makes a token cross every stage, not just one', () => {
    const runs = [1, 2, 4].map((pipeline) =>
      analyseInference({
        ...baseConfig,
        parallelism: { tensor: 2, pipeline, data: 1 },
        batch: 8,
        promptLength: 512,
        generateLength: 16,
      }),
    )

    const [p1, p2, p4] = runs.map((r) => r.timePerOutputToken!)

    // Splitting 80 layers over more stages does not make a single token faster:
    // each stage does less, but the token still visits all of them. Latency
    // stays flat or worsens slightly from the hop cost.
    expect(p2!).toBeGreaterThan(p1! * 0.85)
    expect(p4!).toBeGreaterThan(p1! * 0.85)
  })

  it('shards weights across pipeline stages as well as tensor ranks', () => {
    const one = analyseInference({
      ...baseConfig,
      parallelism: { tensor: 2, pipeline: 1, data: 1 },
      batch: 1,
      promptLength: 512,
      generateLength: 1,
    })
    const four = analyseInference({
      ...baseConfig,
      parallelism: { tensor: 2, pipeline: 4, data: 1 },
      batch: 1,
      promptLength: 512,
      generateLength: 1,
    })

    // Omitting the pipeline divisor from the capacity check falsely rejects
    // configurations that fit comfortably.
    expect(four.capacity.perGpu.weights).toBeLessThan(one.capacity.perGpu.weights / 3)
    expect(four.feasible).toBe(true)
  })
})

describe('infeasible configurations', () => {
  it('refuses to report a time for a model that will not load', () => {
    const result = analyseInference({
      ...baseConfig,
      parallelism: { tensor: 1, pipeline: 1, data: 1 },
      batch: 64,
      promptLength: 8192,
      generateLength: 1,
    })

    // 70B in bf16 is 141 GB of weights against 80 GiB on one card.
    expect(result.feasible).toBe(false)
    expect(result.capacity.verdict).toBe('will-not-fit')
    expect(result.timePerOutputToken).toBeUndefined()
    expect(result.aggregateTokensPerSecond).toBeUndefined()
  })
})
