"""
evaluate.py — Demonstrate and benchmark the Adaptive Context Compression pipeline.

Run:
    uv run python scripts/evaluate.py
"""

from __future__ import annotations

import os
import textwrap
import time

from src.compressor import ContextCompressor


# ──────────────────────────────────────────────────────────────────── #
# Sample long document (simulated research paper, ~2000 tokens)       #
# ──────────────────────────────────────────────────────────────────── #

SAMPLE_DOCUMENT = textwrap.dedent("""\
Title: Advances in Neural Architecture Search for Edge Deployment

Abstract:
Neural Architecture Search (NAS) has become a dominant paradigm for designing deep
learning models. However, most NAS methods target cloud-scale hardware with abundant
compute and memory. This paper addresses the overlooked problem of NAS for edge devices —
microcontrollers, mobile phones, and IoT sensors — where latency, power consumption, and
memory footprint are first-class constraints. We propose EdgeNAS, a hardware-aware search
algorithm that jointly optimizes accuracy, latency, and energy on real target hardware.

1. Introduction

The deployment of neural networks on edge devices presents unique challenges. Unlike
cloud environments, edge hardware has strict constraints on memory (often < 512 KB SRAM),
compute capability (typically ARM Cortex-M class), and energy budget (battery-powered).
Traditional NAS algorithms like DARTS and ENAS produce architectures that are far too large
for such targets.

Recent work has attempted to address this through proxy metrics — estimating latency from
FLOPs or parameter counts. However, these proxies correlate poorly with actual on-device
performance, especially for heterogeneous hardware with varying cache hierarchies and
instruction set capabilities.

Our key contributions are:
1. A hardware-in-the-loop NAS framework that measures real latency during search.
2. A multi-objective Pareto search that balances accuracy, latency, and energy.
3. A novel supernet training strategy tailored for sub-1MB models.
4. Extensive evaluation across three edge platforms: STM32H7, Raspberry Pi Zero, and
   Google Coral Edge TPU.

2. Related Work

Neural Architecture Search was pioneered by Zoph and Le (2017), using reinforcement
learning to discover architectures. Subsequent work introduced weight sharing (ENAS),
differentiable search (DARTS), and evolutionary methods (AmoebaNet). However, these
methods primarily target ImageNet-scale models.

Hardware-aware NAS methods like MnasNet, FBNet, and OFA introduced latency constraints,
but they focus on mobile GPUs (Snapdragon, Mali) rather than microcontrollers. ProxylessNAS
and Once-for-All are closer to our setting but still assume relatively powerful ARM CPUs
with several hundred MB of RAM.

For microcontroller targets, MCUNet introduced a two-stage approach combining architecture
and inference engine co-design. TinyNAS provides a memory-adaptive search space. Our work
extends these ideas by incorporating real-time hardware profiling and energy measurement.

3. Methodology

3.1 Search Space Design

Our search space is specifically designed for memory-constrained targets. We limit the
candidate operators to: depthwise separable convolutions (3x3, 5x5), pointwise convolutions,
inverted residual blocks with expansion ratios {2, 4, 6}, and skip connections. Attention
mechanisms are excluded due to their quadratic memory cost.

The macro architecture follows a linear stack of searchable blocks, each with configurable
depth (1-4 layers), width multiplier (0.25-1.0), and kernel size. Total parameter count is
hard-capped at 500K.

3.2 Hardware-in-the-Loop Profiling

Unlike proxy-based methods, we measure actual inference latency on the target device during
search. We flash each candidate architecture onto the target MCU, run 100 inference passes,
and record the median latency. This adds approximately 8 seconds per evaluation but provides
exact latency measurements rather than estimates.

Energy consumption is measured using an INA219 current sensor placed in series with the
device power supply. We compute energy-per-inference as the integral of power over the
inference duration.

3.3 Multi-Objective Pareto Search

We use NSGA-II with three objectives: validation accuracy (maximize), latency (minimize),
and energy-per-inference (minimize). The algorithm maintains a population of 50 architectures
and runs for 200 generations. Pareto-dominated solutions are pruned each generation.

4. Experimental Results

4.1 Datasets

We evaluate on three standard benchmarks:
- CIFAR-10: 60K images, 10 classes
- Visual Wake Words (VWW): 115K images, 2 classes (person/no-person)
- Google Speech Commands v2: 105K audio samples, 35 keyword classes

4.2 Baselines

We compare against: MobileNetV2 (scaled to fit), MCUNet, TinyNAS, MicroNets, and
manually designed architectures from the MLPerf Tiny benchmark suite.

4.3 Main Results

On CIFAR-10, EdgeNAS achieves 93.2% accuracy with 320KB peak RAM and 18ms latency on
STM32H7, compared to MCUNet's 91.8% at 380KB RAM and 23ms. This represents a 1.4%
accuracy improvement while reducing memory by 16% and latency by 22%.

On Visual Wake Words, EdgeNAS achieves 89.7% accuracy at 11ms latency versus MCUNet's
87.9% at 15ms. The energy-per-inference is 0.42mJ compared to 0.58mJ.

On Speech Commands, EdgeNAS achieves 94.1% accuracy on the 12-class subset, matching
DS-CNN performance while reducing model size by 3.2x.

4.4 Ablation Study

We ablate the contribution of each component:
- Removing hardware-in-the-loop: accuracy drops 0.8%, latency prediction error increases 35%
- Using single-objective (accuracy only): 2.1x higher latency at equivalent accuracy
- Without supernet weight sharing: search time increases from 18 GPU-hours to 400+ GPU-hours

5. Limitations

Several limitations should be acknowledged:
- The hardware-in-the-loop approach requires physical access to the target device, limiting
  scalability to new hardware platforms.
- Search cost, while improved, is still significant: ~18 GPU-hours plus ~6 hours of device
  profiling for each target platform.
- Our search space deliberately excludes attention mechanisms, which may limit performance
  on sequence tasks.
- The energy measurements are averaged over steady-state inference and do not account for
  cold-start effects or dynamic voltage/frequency scaling.
- Results on larger datasets (ImageNet) are not yet available.

6. Conclusion

We presented EdgeNAS, a hardware-aware neural architecture search framework for edge
devices. By incorporating real hardware measurements into the search loop, we achieve
architectures that are Pareto-optimal across accuracy, latency, and energy. Our method
outperforms existing approaches on three benchmarks across three hardware platforms.

Future work includes extending the framework to support transformer-based architectures
with linear attention, implementing on-device training capabilities, and building a
hardware cost model that can generalise across device families without requiring physical
profiling.

References

[1] Zoph, B., & Le, Q. V. (2017). Neural architecture search with reinforcement learning.
[2] Liu, H., Simonyan, K., & Yang, Y. (2019). DARTS: Differentiable Architecture Search.
[3] Cai, H., Zhu, L., & Han, S. (2020). ProxylessNAS.
[4] Lin, J., Chen, W., Lin, Y., et al. (2020). MCUNet.
[5] Tan, M., et al. (2019). MnasNet: Platform-Aware Neural Architecture Search.
""")

SAMPLE_QUERY = "What are the main contributions and limitations of this work?"


# ──────────────────────────────────────────────────────────────────── #
# Evaluation helpers                                                   #
# ──────────────────────────────────────────────────────────────────── #


def print_banner(text: str) -> None:
    width = 72
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


def run_evaluation() -> None:
    print_banner("Adaptive Context Compression — V1 Evaluation")

    print(f"\n📄 Document length : {len(SAMPLE_DOCUMENT):,} characters")
    print(f"🔍 Query           : {SAMPLE_QUERY!r}\n")

    # --- initialise (loads embedding model, may take a few seconds) ---
    print("⏳ Initialising pipeline (loading embedding model)...")
    t0 = time.perf_counter()
    compressor = ContextCompressor(
        max_tokens=1500,
        groq_api_key=os.environ.get("GROQ_API_KEY"),
    )
    init_time = time.perf_counter() - t0
    print(f"   ✅ Ready in {init_time:.1f}s\n")

    # --- compress ---
    print("⏳ Compressing...")
    t0 = time.perf_counter()
    result = compressor.compress(document=SAMPLE_DOCUMENT, query=SAMPLE_QUERY)
    compress_time = time.perf_counter() - t0

    # --- report ---
    print_banner("Results")

    print(f"  Original tokens   : {result.original_tokens:,}")
    print(f"  Compressed tokens : {result.compressed_tokens:,}")
    print(f"  Token reduction   : {result.reduction_pct:.1f}%")
    print(f"  Chunks created    : {result.num_chunks}")
    print(f"  Tier distribution : {result.tier_counts}")
    print(f"  Compression time  : {compress_time:.3f}s")

    print_banner("Compressed Context (preview)")
    preview = result.compressed_text[:2000]
    print(preview)
    if len(result.compressed_text) > 2000:
        print(f"\n  ... [{len(result.compressed_text) - 2000} more chars]")

    print_banner("Done ✅")


if __name__ == "__main__":
    run_evaluation()
