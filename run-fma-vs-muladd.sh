#!/usr/bin/env bash
# Build and run the FMA vs MUL+ADD comparison demos.
# Requires JDK 21+.
#
# For PrintAssembly output you also need hsdis on the JVM library path:
#   Linux:  apt install libhsdis0-jvmci  (or build hsdis from OpenJDK source)
#   macOS:  place hsdis-aarch64.dylib (or hsdis-amd64.dylib) in $JAVA_HOME/lib/server/
#
# What to look for in the generated assembly:
#   AMD64 AVX2:  FMA     -> vfmadd231pd ymm   (4 doubles, one instruction)
#                MUL+ADD -> vmulpd ymm + vaddpd ymm
#   AArch64:     FMA     -> fmla v0.2d, v1.2d, v2.2d
#                MUL+ADD -> fmul v0.2d, v1.2d, v2.2d  +  fadd v0.2d, v0.2d, v3.2d
#                           (MUL+ADD may be faster on Apple M-series -- see FmaVsMulAdd.java)

set -euo pipefail
cd "$(dirname "$0")"

FLAGS="--add-modules jdk.incubator.vector"
ASM_FLAGS="-XX:+UnlockDiagnosticVMOptions -XX:+PrintAssembly -XX:-TieredCompilation -XX:CompileThreshold=50"
SRCDIR="src/main/java/org/przybyl/simd/fmavsmuladd"

echo "=== Compiling ==="
mkdir -p out/fma out/muladd out/combined
javac $FLAGS -d out/fma      "$SRCDIR/OperationsFMA.java"
javac $FLAGS -d out/muladd   "$SRCDIR/OperationsMULADD.java"
javac $FLAGS -d out/combined "$SRCDIR/FmaVsMulAdd.java"

echo ""
echo "=== OperationsFMA (va.fma(vb, sum)) ==="
java $FLAGS -cp out/fma org.przybyl.simd.fmavsmuladd.OperationsFMA

echo ""
echo "=== OperationsMULADD (sum.add(va.mul(vb))) ==="
java $FLAGS -cp out/muladd org.przybyl.simd.fmavsmuladd.OperationsMULADD

echo ""
echo "=== FmaVsMulAdd (combined side-by-side comparison) ==="
java $FLAGS -XX:CompileThreshold=50 -cp out/combined org.przybyl.simd.fmavsmuladd.FmaVsMulAdd

# PrintAssembly sections -- require hsdis
if [[ "${1:-}" == "--asm" ]]; then
    echo ""
    echo "=== PrintAssembly: dotFMA ==="
    echo "    AMD64:   look for vfmadd231pd on ymm (AVX2) or zmm (AVX-512) registers"
    echo "    AArch64: look for fmla v0.2d (or similar NEON FMA instruction)"
    java $FLAGS $ASM_FLAGS \
         -XX:CompileOnly="FmaVsMulAdd::dotFMA" \
         -cp out/combined org.przybyl.simd.fmavsmuladd.FmaVsMulAdd 2>&1 | \
         awk '/Compiled method.*FmaVsMulAdd::dotFMA/{show=1} show{print} show && /\[\/MachCode\]/{show=0}' || true

    echo ""
    echo "=== PrintAssembly: dotMulAdd ==="
    echo "    AMD64:   look for vmulpd then vaddpd on ymm/zmm registers"
    echo "    AArch64: look for fmul then fadd (or fmla if the JIT fuses them)"
    java $FLAGS $ASM_FLAGS \
         -XX:CompileOnly="FmaVsMulAdd::dotMulAdd" \
         -cp out/combined org.przybyl.simd.fmavsmuladd.FmaVsMulAdd 2>&1 | \
         awk '/Compiled method.*FmaVsMulAdd::dotMulAdd/{show=1} show{print} show && /\[\/MachCode\]/{show=0}' || true
else
    echo ""
    echo "Tip: run with --asm to also print the generated assembly (requires hsdis)."
fi
