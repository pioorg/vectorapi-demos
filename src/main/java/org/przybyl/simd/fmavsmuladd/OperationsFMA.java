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

/// Please remember to run with a VM option
/// `--add-modules jdk.incubator.vector`
/// For more details, please see [JEP-529](https://openjdk.org/jeps/529).
@SuppressWarnings("DuplicatedCode")
public class OperationsFMA {
    public static void main(String[] args) {
        int size = 1_000_000;
        double[] a = new double[size];
        double[] b = new double[size];
        double[] c = new double[size];

        // Initialize arrays with deterministic values (avoids Math.random() for reproducibility).
        for (int i = 0; i < size; i++) {
            a[i] = (i + 1) * 0.000_001;
            b[i] = (size - i) * 0.000_001;
        }

        MathOps scalar = new ScalarOps();
        MathOps vector = new VectorFMAOps();

        // Warm-up phase for JIT compilation.
        for (int i = 0; i < 5; i++) {
            scalar.dotProduct(a, b);
            scalar.add(a, b, c);
            scalar.euclideanNorm(a);
            vector.dotProduct(a, b);
            vector.add(a, b, c);
            vector.euclideanNorm(a);
        }

        // Benchmark dot product.
        long start = System.nanoTime();
        double scalarDot = scalar.dotProduct(a, b);
        long scalarDotTime = System.nanoTime() - start;
        System.out.printf("Scalar Dot Product: %f, Time: %.3f ms%n", scalarDot, scalarDotTime / 1e6);

        start = System.nanoTime();
        double vectorDot = vector.dotProduct(a, b);
        long vectorDotTime = System.nanoTime() - start;
        System.out.printf("Vector Dot Product (FMA): %f, Time: %.3f ms%n", vectorDot, vectorDotTime / 1e6);
        System.out.printf("Speedup (Dot): %.2f×%n", (double) scalarDotTime / vectorDotTime);

        // Benchmark element-wise addition.
        start = System.nanoTime();
        scalar.add(a, b, c);
        long scalarAddTime = System.nanoTime() - start;
        System.out.printf("Scalar Addition Time: %.3f ms%n", scalarAddTime / 1e6);

        start = System.nanoTime();
        vector.add(a, b, c);
        long vectorAddTime = System.nanoTime() - start;
        System.out.printf("Vector Addition Time: %.3f ms%n", vectorAddTime / 1e6);
        System.out.printf("Speedup (Addition): %.2f×%n", (double) scalarAddTime / vectorAddTime);

        // Benchmark Euclidean norm.
        start = System.nanoTime();
        double scalarNorm = scalar.euclideanNorm(a);
        long scalarNormTime = System.nanoTime() - start;
        System.out.printf("Scalar Euclidean Norm: %f, Time: %.3f ms%n", scalarNorm, scalarNormTime / 1e6);

        start = System.nanoTime();
        double vectorNorm = vector.euclideanNorm(a);
        long vectorNormTime = System.nanoTime() - start;
        System.out.printf("Vector Euclidean Norm (FMA): %f, Time: %.3f ms%n", vectorNorm, vectorNormTime / 1e6);
        System.out.printf("Speedup (Norm): %.2f×%n", (double) scalarNormTime / vectorNormTime);
    }
}


class VectorFMAOps implements MathOps {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    /// `va.fma(vb, sum)` -- fused multiply-add, one rounding per step.
    ///
    /// - AMD64 AVX2: `VFMADD231PD ymm` (4 doubles at once)
    /// - AArch64: `FMLA v0.2d, v1.2d, v2.2d`
    public double dotProduct(double[] a, double[] b) {
        int i = 0;
        int upperBound = SPECIES.loopBound(a.length);
        var sum = DoubleVector.zero(SPECIES);
        for (; i < upperBound; i += SPECIES.length()) {
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

    public void add(double[] a, double[] b, double[] c) {
        int i = 0;
        int upperBound = SPECIES.loopBound(a.length);
        for (; i < upperBound; i += SPECIES.length()) {
            var va = DoubleVector.fromArray(SPECIES, a, i);
            var vb = DoubleVector.fromArray(SPECIES, b, i);
            va.add(vb).intoArray(c, i);
        }
        for (; i < a.length; i++) {
            c[i] = a[i] + b[i];
        }
    }

    public double euclideanNorm(double[] a) {
        int i = 0;
        int upperBound = SPECIES.loopBound(a.length);
        var sum = DoubleVector.zero(SPECIES);
        for (; i < upperBound; i += SPECIES.length()) {
            var va = DoubleVector.fromArray(SPECIES, a, i);
            sum = va.fma(va, sum);
        }
        double result = sum.reduceLanes(VectorOperators.ADD);
        for (; i < a.length; i++) {
            result += a[i] * a[i];
        }
        return Math.sqrt(result);
    }
}
