#!/usr/bin/env bash
# Compile and run the dot product demo.
# Requires JDK 21+.

set -euo pipefail
cd "$(dirname "$0")"

FLAGS="--add-modules jdk.incubator.vector"
SRC="src/main/java/org/przybyl/simd/dotproduct/DotProductDemo.java"
OUT="out/dotproduct"

mkdir -p "$OUT"

echo "=== Compiling ==="
javac $FLAGS -d "$OUT" "$SRC"

echo ""
echo "=== Plain run ==="
java $FLAGS -cp "$OUT" org.przybyl.simd.dotproduct.DotProductDemo

echo ""
echo "=== Run with PrintAssembly (needs hsdis on the JVM library path) ==="
echo "    Look for VFMADD231PS in dotProductVector and VMULSS/VADDSS in dotProductScalar."
echo ""
java $FLAGS \
     -XX:+UnlockDiagnosticVMOptions \
     -XX:+PrintAssembly \
     -XX:CompileOnly="DotProductDemo::dotProductScalar,DotProductDemo::dotProductVector" \
     -XX:-TieredCompilation \
     -cp "$OUT" org.przybyl.simd.dotproduct.DotProductDemo 2>&1 | grep -A 40 "dotProduct" || true
