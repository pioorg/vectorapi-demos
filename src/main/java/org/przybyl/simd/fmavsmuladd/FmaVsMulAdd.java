/*
 * Copyright 2026 Piotr Przybył
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

package org.przybyl.simd.fmavsmuladd;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/// Side-by-side benchmark: FMA vs MUL+ADD for vector dot product.
///
/// Two Vector API strategies that produce the same mathematical result
/// but different floating-point results (different rounding) and can
/// have different performance depending on the CPU:
///
/// - **FMA:** `va.fma(vb, sum)` -- one fused multiply-add, one rounding
/// - **MUL+ADD:** `sum.add(va.mul(vb))` -- separate multiply then add, two roundings
///
/// ## What the JIT emits
///
/// **AMD64 with AVX2:**
/// - FMA: `VFMADD231PD ymm` (4 doubles, one instruction)
/// - MUL+ADD: `VMULPD ymm` + `VADDPD ymm` (two instructions) -- FMA typically wins.
///
/// **AArch64 (Apple M-series, NEON):**
/// - FMA: `FMLA v0.2d, v1.2d, v2.2d`
/// - MUL+ADD: `FMUL v0.2d, v1.2d, v2.2d` + `FADD v0.2d, v0.2d, v3.2d`
/// - MUL+ADD can be **faster** on Apple Silicon: `FMUL` does not read the
///   accumulator, so the CPU can issue the next `FMUL` while the previous
///   `FADD` is still in flight. `FMLA` must wait for the accumulator from the
///   previous iteration. With equal-throughput multiply and add units,
///   MUL+ADD hides the add latency; FMA cannot.
///
/// To capture the assembly, run with PrintAssembly (see `run-fma-vs-muladd.sh`):
/// - AMD64: `grep -E 'vfmadd|vmulpd|vaddpd'`
/// - AArch64: `grep -E 'fmla|fmul|fadd'`
@SuppressWarnings("DuplicatedCode")
public class FmaVsMulAdd {

    static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    /// Strategy 1: fused multiply-add -- one rounding per iteration.
    static double dotFMA(double[] a, double[] b) {
        int i = 0;
        int bound = SPECIES.loopBound(a.length);
        var sum = DoubleVector.zero(SPECIES);
        for (; i < bound; i += SPECIES.length()) {
            var va = DoubleVector.fromArray(SPECIES, a, i);
            var vb = DoubleVector.fromArray(SPECIES, b, i);
            sum = va.fma(vb, sum);
        }
        double result = sum.reduceLanes(VectorOperators.ADD);
        for (; i < a.length; i++) {
            result += a[i] * b[i];
        }
        return result;
    }

    /// Strategy 2: separate multiply then add -- two roundings per iteration.
    static double dotMulAdd(double[] a, double[] b) {
        int i = 0;
        int bound = SPECIES.loopBound(a.length);
        var sum = DoubleVector.zero(SPECIES);
        for (; i < bound; i += SPECIES.length()) {
            var va = DoubleVector.fromArray(SPECIES, a, i);
            var vb = DoubleVector.fromArray(SPECIES, b, i);
            sum = sum.add(va.mul(vb));
        }
        double result = sum.reduceLanes(VectorOperators.ADD);
        for (; i < a.length; i++) {
            result += a[i] * b[i];
        }
        return result;
    }

    public static void main(String[] args) {
        int size = 100_000;
        double[] a = new double[size];
        double[] b = new double[size];
        for (int i = 0; i < size; i++) {
            a[i] = (i + 1) * 0.000_001;
            b[i] = (size - i) * 0.000_001;
        }

        System.out.printf("SPECIES_PREFERRED: %s (%d doubles per vector register)%n",
                SPECIES, SPECIES.length());

        double sink = 0;
        for (int w = 0; w < 50; w++) {
            sink += dotFMA(a, b);
            sink += dotMulAdd(a, b);
        }
        System.out.println("Warm-up done (sink=" + sink + ")");

        int RUNS = 50;

        long fmaTotal = 0;
        double fmaResult = 0;
        for (int r = 0; r < RUNS; r++) {
            long t = System.nanoTime();
            fmaResult = dotFMA(a, b);
            fmaTotal += System.nanoTime() - t;
        }

        long mulAddTotal = 0;
        double mulAddResult = 0;
        for (int r = 0; r < RUNS; r++) {
            long t = System.nanoTime();
            mulAddResult = dotMulAdd(a, b);
            mulAddTotal += System.nanoTime() - t;
        }

        double fmaMs     = fmaTotal    / 1e6 / RUNS;
        double mulAddMs  = mulAddTotal / 1e6 / RUNS;

        System.out.println();
        System.out.printf("FMA      result=%.6f  avg=%.3f ms%n", fmaResult, fmaMs);
        System.out.printf("MUL+ADD  result=%.6f  avg=%.3f ms%n", mulAddResult, mulAddMs);
        System.out.printf("Ratio FMA/MUL+ADD: %.2f  (>1 means FMA is faster)%n", mulAddMs / fmaMs);
        System.out.println();
        System.out.println("Results differ because FMA has one rounding; MUL+ADD has two.");
        System.out.println("Intel/AMD AVX2: FMA usually wins (one instruction vs two).");
        System.out.println("Apple Silicon M-series: MUL+ADD often wins (overlapping mul/add units).");
    }
}
