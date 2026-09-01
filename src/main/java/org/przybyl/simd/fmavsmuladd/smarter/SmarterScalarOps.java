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

import org.przybyl.simd.fmavsmuladd.MathOps;

@SuppressWarnings("DuplicatedCode")
class SmarterScalarOps implements MathOps {
    public double dotProduct(double[] a, double[] b) {
        double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int i = 0;
        for (; i <= a.length - 4; i += 4) {
            s0 += a[i] * b[i];
            s1 += a[i + 1] * b[i + 1];
            s2 += a[i + 2] * b[i + 2];
            s3 += a[i + 3] * b[i + 3];
        }
        double sum = s0 + s1 + s2 + s3;
        for (; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    public void add(double[] a, double[] b, double[] c) {
        int i = 0;
        for (; i <= a.length - 4; i += 4) {
            c[i] = a[i] + b[i];
            c[i + 1] = a[i + 1] + b[i + 1];
            c[i + 2] = a[i + 2] + b[i + 2];
            c[i + 3] = a[i + 3] + b[i + 3];
        }
        for (; i < a.length; i++) {
            c[i] = a[i] + b[i];
        }
    }

    public double euclideanNorm(double[] a) {
        double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int i = 0;
        for (; i <= a.length - 4; i += 4) {
            s0 += a[i] * a[i];
            s1 += a[i + 1] * a[i + 1];
            s2 += a[i + 2] * a[i + 2];
            s3 += a[i + 3] * a[i + 3];
        }
        double sum = s0 + s1 + s2 + s3;
        for (; i < a.length; i++) {
            sum += a[i] * a[i];
        }
        return Math.sqrt(sum);
    }
}
