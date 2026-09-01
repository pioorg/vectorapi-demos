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

class ScalarOps implements MathOps {
    public double dotProduct(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    public void add(double[] a, double[] b, double[] c) {
        for (int i = 0; i < a.length; i++) {
            c[i] = a[i] + b[i];
        }
    }

    public double euclideanNorm(double[] a) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * a[i];
        }
        return Math.sqrt(sum);
    }
}
