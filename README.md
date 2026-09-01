# Demos

Standalone Java demos for the Piotr's SIMD talk. Requires JDK 25+.

## Structure

```
src/main/java/org/przybyl/simd/
  dotproduct/      DotProductDemo      -- scalar vs Vector API float dot product
  fmavsmuladd/     FmaVsMulAdd         -- side-by-side FMA vs MUL+ADD timing
                   OperationsFMA       -- Vector API with va.fma(vb, sum)
                   OperationsMULADD    -- Vector API with sum.add(va.mul(vb))
```

## Building with Maven

```bash
mvn compile
```

Requires Java 25. The compiler plugin passes `--add-modules jdk.incubator.vector` automatically.

## Running

Each demo has a shell script at the project root. The scripts compile on the fly with `javac` (no Maven needed) and run immediately.

### dot-product

Scalar vs explicit Vector API float dot product on 768-element arrays (embedding model dimension).
Shows the 4-accumulator unrolling pattern used by Lucene/Elasticsearch.

```bash
./run-dot-product.sh
```

### fma-vs-muladd

Compares two Vector API dot-product strategies on 1M-element double arrays.

```bash
./run-fma-vs-muladd.sh           # timing only
./run-fma-vs-muladd.sh --asm     # timing + PrintAssembly (requires hsdis)
```

Assembly output requires `hsdis` on the JVM library path.

## Capturing output

Run these snippets once per machine. Output lands in `output/`.

### Apple M* (AArch64 NEON)

```bash
./run-dot-product.sh    > output/dot-product-aarch64-m3.txt    2>&1
./run-fma-vs-muladd.sh --asm > output/fma-vs-muladd-aarch64-m3.txt 2>&1
```

### Linux amd64 (x86-64 AVX2)

```bash
./run-dot-product.sh    > output/dot-product-amd64-avx2.txt    2>&1
./run-fma-vs-muladd.sh --asm > output/fma-vs-muladd-amd64-avx2.txt 2>&1
```

## What to look for in the assembly

| Architecture | FMA | MUL+ADD |
|---|---|---|
| AMD64 AVX2 | `vfmadd231pd ymm` | `vmulpd ymm` + `vaddpd ymm` |
| AArch64 NEON | `fmla v0.2d` | `fmul v0.2d` + `fadd v0.2d` |

On Intel/AMD, FMA is one instruction vs two -- typically equal or slightly faster.
On Apple Silicon M-series, MUL+ADD is often faster: `fmul` does not read the accumulator,
so the CPU can issue the next multiply while the previous add is still in flight.
