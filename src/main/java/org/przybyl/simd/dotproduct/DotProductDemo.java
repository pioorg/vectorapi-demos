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

package org.przybyl.simd.dotproduct;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/// Demonstrates scalar vs Vector API float dot product.
///
/// To see the assembly C2 generates, run with:
/// ```
/// java --add-modules jdk.incubator.vector \
///      -XX:+UnlockDiagnosticVMOptions \
///      -XX:+PrintAssembly \
///      -XX:CompileOnly=DotProductDemo::dotProductScalar,DotProductDemo::dotProductVector \
///      -XX:-TieredCompilation \
///      DotProductDemo
/// ```
///
/// Requires `hsdis` (disassembler plugin) on the JVM library path.
///
/// What to look for:
/// - AVX2: `VFMADD231PS ymm` (8 floats at once)
/// - AVX-512: `VFMADD231PS zmm` (16 floats at once)
/// - Scalar path: `VMULSS` + `VADDSS` on `xmm` (1 float at a time)
public class DotProductDemo {

    static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    static float dotProductScalar(float[] a, float[] b) {
        float sum = 0f;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    static float dotProductVector(float[] a, float[] b) {
        int i = 0;
        int bound = SPECIES.loopBound(a.length);

        var acc0 = FloatVector.zero(SPECIES);
        var acc1 = FloatVector.zero(SPECIES);
        var acc2 = FloatVector.zero(SPECIES);
        var acc3 = FloatVector.zero(SPECIES);
        int step = SPECIES.length();

        for (; i < bound - 3 * step; i += 4 * step) {
            acc0 = FloatVector.fromArray(SPECIES, a, i)            .fma(FloatVector.fromArray(SPECIES, b, i),            acc0);
            acc1 = FloatVector.fromArray(SPECIES, a, i +     step) .fma(FloatVector.fromArray(SPECIES, b, i +     step), acc1);
            acc2 = FloatVector.fromArray(SPECIES, a, i + 2 * step) .fma(FloatVector.fromArray(SPECIES, b, i + 2 * step), acc2);
            acc3 = FloatVector.fromArray(SPECIES, a, i + 3 * step) .fma(FloatVector.fromArray(SPECIES, b, i + 3 * step), acc3);
        }
        // remaining full vectors
        for (; i < bound; i += step) {
            acc0 = FloatVector.fromArray(SPECIES, a, i).fma(FloatVector.fromArray(SPECIES, b, i), acc0);
        }
        float result = acc0.add(acc1).add(acc2).add(acc3).reduceLanes(VectorOperators.ADD);
        // scalar tail
        for (; i < a.length; i++) {
            result += a[i] * b[i];
        }
        return result;
    }

    public static void main(String[] args) {
        int dims = 768;
        float[] a = new float[dims];
        float[] b = new float[dims];
        for (int i = 0; i < dims; i++) {
            a[i] = (i + 1) * 0.001f;
            b[i] = (dims - i) * 0.001f;
        }

        System.out.printf("SPECIES_PREFERRED: %s (%d floats per instruction)%n",
                SPECIES, SPECIES.length());

        // warm up: trigger C2 compilation
        float sink = 0f;
        for (int w = 0; w < 20_000; w++) {
            sink += dotProductScalar(a, b);
            sink += dotProductVector(a, b);
        }
        System.out.println("Warm-up done (sink=" + sink + ")");

        // timed runs
        int RUNS = 500_000;

        long t0 = System.nanoTime();
        for (int r = 0; r < RUNS; r++) sink += dotProductScalar(a, b);
        long scalarNs = System.nanoTime() - t0;

        long t1 = System.nanoTime();
        for (int r = 0; r < RUNS; r++) sink += dotProductVector(a, b);
        long vectorNs = System.nanoTime() - t1;

        System.out.printf("Scalar:     %,d ns total  (%,.1f ns/call)%n", scalarNs, (double) scalarNs / RUNS);
        System.out.printf("Vector API: %,d ns total  (%,.1f ns/call)%n", vectorNs, (double) vectorNs / RUNS);
        System.out.printf("Speedup:    %.2fx%n", (double) scalarNs / vectorNs);
        System.out.println("(sink=" + sink + ")");
    }
}
