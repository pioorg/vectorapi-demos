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

package org.przybyl.simd.fmavsmuladd.smarter;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.przybyl.simd.fmavsmuladd.MathOps;

@SuppressWarnings("DuplicatedCode")
class SmarterVectorFMAOps implements MathOps {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    public double dotProduct(double[] a, double[] b) {
        int i = 0;
        int bound = SPECIES.loopBound(a.length);
        int step = SPECIES.length();

        var acc0 = DoubleVector.zero(SPECIES);
        var acc1 = DoubleVector.zero(SPECIES);
        var acc2 = DoubleVector.zero(SPECIES);
        var acc3 = DoubleVector.zero(SPECIES);

        for (; i <= bound - 4 * step; i += 4 * step) {
            acc0 = DoubleVector.fromArray(SPECIES, a, i).fma(DoubleVector.fromArray(SPECIES, b, i), acc0);
            acc1 = DoubleVector.fromArray(SPECIES, a, i + step).fma(DoubleVector.fromArray(SPECIES, b, i + step), acc1);
            acc2 = DoubleVector.fromArray(SPECIES, a, i + 2 * step).fma(DoubleVector.fromArray(SPECIES, b, i + 2 * step), acc2);
            acc3 = DoubleVector.fromArray(SPECIES, a, i + 3 * step).fma(DoubleVector.fromArray(SPECIES, b, i + 3 * step), acc3);
        }
        for (; i < bound; i += step) {
            acc0 = DoubleVector.fromArray(SPECIES, a, i).fma(DoubleVector.fromArray(SPECIES, b, i), acc0);
        }
        double result = acc0.add(acc1).add(acc2).add(acc3).reduceLanes(VectorOperators.ADD);
        for (; i < a.length; i++) {
            result += a[i] * b[i];
        }
        return result;
    }

    public void add(double[] a, double[] b, double[] c) {
        int i = 0;
        int bound = SPECIES.loopBound(a.length);
        int step = SPECIES.length();

        for (; i <= bound - 4 * step; i += 4 * step) {
            DoubleVector.fromArray(SPECIES, a, i).add(DoubleVector.fromArray(SPECIES, b, i)).intoArray(c, i);
            DoubleVector.fromArray(SPECIES, a, i + step).add(DoubleVector.fromArray(SPECIES, b, i + step)).intoArray(c, i + step);
            DoubleVector.fromArray(SPECIES, a, i + 2 * step).add(DoubleVector.fromArray(SPECIES, b, i + 2 * step)).intoArray(c, i + 2 * step);
            DoubleVector.fromArray(SPECIES, a, i + 3 * step).add(DoubleVector.fromArray(SPECIES, b, i + 3 * step)).intoArray(c, i + 3 * step);
        }
        for (; i < bound; i += step) {
            DoubleVector.fromArray(SPECIES, a, i).add(DoubleVector.fromArray(SPECIES, b, i)).intoArray(c, i);
        }
        for (; i < a.length; i++) {
            c[i] = a[i] + b[i];
        }
    }

    public double euclideanNorm(double[] a) {
        int i = 0;
        int bound = SPECIES.loopBound(a.length);
        int step = SPECIES.length();

        var acc0 = DoubleVector.zero(SPECIES);
        var acc1 = DoubleVector.zero(SPECIES);
        var acc2 = DoubleVector.zero(SPECIES);
        var acc3 = DoubleVector.zero(SPECIES);

        for (; i <= bound - 4 * step; i += 4 * step) {
            acc0 = DoubleVector.fromArray(SPECIES, a, i).fma(DoubleVector.fromArray(SPECIES, a, i), acc0);
            acc1 = DoubleVector.fromArray(SPECIES, a, i + step).fma(DoubleVector.fromArray(SPECIES, a, i + step), acc1);
            acc2 = DoubleVector.fromArray(SPECIES, a, i + 2 * step).fma(DoubleVector.fromArray(SPECIES, a, i + 2 * step), acc2);
            acc3 = DoubleVector.fromArray(SPECIES, a, i + 3 * step).fma(DoubleVector.fromArray(SPECIES, a, i + 3 * step), acc3);
        }
        for (; i < bound; i += step) {
            acc0 = DoubleVector.fromArray(SPECIES, a, i).fma(DoubleVector.fromArray(SPECIES, a, i), acc0);
        }
        double result = acc0.add(acc1).add(acc2).add(acc3).reduceLanes(VectorOperators.ADD);
        for (; i < a.length; i++) {
            result += a[i] * a[i];
        }
        return Math.sqrt(result);
    }
}
