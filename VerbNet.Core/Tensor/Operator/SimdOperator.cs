using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace VerbNet.Core
{
    public static unsafe class SimdOperator
    {
        public const int AVX_VECTOR_SIZE = 8;
        public const int BLOCK_SIZE = 4096;
        public const int MATRIX_BLOCK_SIZE = 32;

        private static int _maxDegreeOfParallelism => Environment.ProcessorCount;
        private static ParallelOptions _parallelOptions = new ParallelOptions
        {
            MaxDegreeOfParallelism = _maxDegreeOfParallelism
        };
        private static Vector256<float> _zeroVec = Vector256<float>.Zero;
        private static Vector256<float> _oneVec = Vector256.Create(1f);
        private static Vector256<float> _minusOneVec = Vector256.Create(-1f);

        public static void Add(float* a, float* b, float* result, int length)
        {
            Parallel.For(0, (length + BLOCK_SIZE - 1) / BLOCK_SIZE, _parallelOptions, chunkIndex =>
            {
                int start = chunkIndex * BLOCK_SIZE;
                int end = Math.Min(start + BLOCK_SIZE, length);

                int i = start;
                int vectorizableLength = end - start;
                int vectorizedEnd = start + (vectorizableLength / AVX_VECTOR_SIZE) * AVX_VECTOR_SIZE;

                for (; i < vectorizedEnd; i += AVX_VECTOR_SIZE)
                {
                    Vector256<float> aVec = Avx.LoadAlignedVector256(a + i);
                    Vector256<float> bVec = Avx.LoadAlignedVector256(b + i);
                    Vector256<float> resultVec = Avx.Add(aVec, bVec);
                    Avx.StoreAligned(result + i, resultVec);
                }
                for (; i < end; i++)
                {
                    result[i] = a[i] + b[i];
                }
            });
        }

        public static void Subtract(float* a, float* b, float* result, int length)
        {
            Parallel.For(0, (length + BLOCK_SIZE - 1) / BLOCK_SIZE, _parallelOptions, chunkIndex =>
            {
                int start = chunkIndex * BLOCK_SIZE;
                int end = Math.Min(start + BLOCK_SIZE, length);

                int i = start;
                int vectorizableLength = end - start;
                int vectorizedEnd = start + (vectorizableLength / AVX_VECTOR_SIZE) * AVX_VECTOR_SIZE;

                for (; i < vectorizedEnd; i += AVX_VECTOR_SIZE)
                {
                    Vector256<float> aVec = Avx.LoadAlignedVector256(a + i);
                    Vector256<float> bVec = Avx.LoadAlignedVector256(b + i);
                    Vector256<float> resultVec = Avx.Subtract(aVec, bVec);
                    Avx.StoreAligned(result + i, resultVec);
                }
                for (; i < end; i++)
                {
                    result[i] = a[i] - b[i];
                }
            });
        }

        public static void Multiply(float* a, float* b, float* result, int length)
        {
            Parallel.For(0, (length + BLOCK_SIZE - 1) / BLOCK_SIZE, _parallelOptions, chunkIndex =>
            {
                int start = chunkIndex * BLOCK_SIZE;
                int end = Math.Min(start + BLOCK_SIZE, length);

                int i = start;
                int vectorizableLength = end - start;
                int vectorizedEnd = start + (vectorizableLength / AVX_VECTOR_SIZE) * AVX_VECTOR_SIZE;

                for (; i < vectorizedEnd; i += AVX_VECTOR_SIZE)
                {
                    Vector256<float> aVec = Avx.LoadAlignedVector256(a + i);
                    Vector256<float> bVec = Avx.LoadAlignedVector256(b + i);
                    Vector256<float> resultVec = Avx.Multiply(aVec, bVec);
                    Avx.StoreAligned(result + i, resultVec);
                }
                for (; i < end; i++)
                {
                    result[i] = a[i] * b[i];
                }
            });
        }

        public static void Divide(float* a, float* b, float* result, int length)
        {
            Parallel.For(0, (length + BLOCK_SIZE - 1) / BLOCK_SIZE, _parallelOptions, chunkIndex =>
            {
                int start = chunkIndex * BLOCK_SIZE;
                int end = Math.Min(start + BLOCK_SIZE, length);

                int i = start;
                int vectorizableLength = end - start;
                int vectorizedEnd = start + (vectorizableLength / AVX_VECTOR_SIZE) * AVX_VECTOR_SIZE;

                for (; i < vectorizedEnd; i += AVX_VECTOR_SIZE)
                {
                    Vector256<float> aVec = Avx.LoadAlignedVector256(a + i);
                    Vector256<float> bVec = Avx.LoadAlignedVector256(b + i);
                    Vector256<float> resultVec = Avx.Divide(aVec, bVec);
                    Avx.StoreAligned(result + i, resultVec);
                }
                for (; i < end; i++)
                {
                    result[i] = a[i] / b[i];
                }
            });
        }

        public static void Negate(float* a, float* result, int length)
        {
            Parallel.For(0, (length + BLOCK_SIZE - 1) / BLOCK_SIZE, _parallelOptions, chunkIndex =>
            {
                int start = chunkIndex * BLOCK_SIZE;
                int end = Math.Min(start + BLOCK_SIZE, length);

                int i = start;
                int vectorizableLength = end - start;
                int vectorizedEnd = start + (vectorizableLength / AVX_VECTOR_SIZE) * AVX_VECTOR_SIZE;

                for (; i < vectorizedEnd; i += AVX_VECTOR_SIZE)
                {
                    Vector256<float> aVec = Avx.LoadAlignedVector256(a + i);
                    Vector256<float> resultVec = Avx.Subtract(_zeroVec, aVec);
                    Avx.StoreAligned(result + i, resultVec);
                }
                for (; i < end; i++)
                {
                    result[i] = -a[i];
                }
            });
        }

        public static void Abs(float* a, float* result, int length)
        {
            Parallel.For(0, (length + BLOCK_SIZE - 1) / BLOCK_SIZE, _parallelOptions, chunkIndex =>
            {
                int start = chunkIndex * BLOCK_SIZE;
                int end = Math.Min(start + BLOCK_SIZE, length);

                int i = start;
                int vectorizableLength = end - start;
                int vectorizedEnd = start + (vectorizableLength / AVX_VECTOR_SIZE) * AVX_VECTOR_SIZE;

                for (; i < vectorizedEnd; i += AVX_VECTOR_SIZE)
                {
                    Vector256<float> aVec = Avx.LoadAlignedVector256(a + i);
                    Vector256<float> mask = Avx.CompareLessThan(aVec, _zeroVec);
                    Vector256<float> resultVec = Avx.Xor(aVec, mask);
                    resultVec = Avx.Subtract(resultVec, mask);
                    Avx.StoreAligned(result + i, resultVec);
                }
                for (; i < end; i++)
                {
                    result[i] = MathF.Abs(a[i]);
                }
            });
        }

        public static void Sign(float* a, float* result, int length)
        {
            Parallel.For(0, (length + BLOCK_SIZE - 1) / BLOCK_SIZE, _parallelOptions, chunkIndex =>
            {
                int start = chunkIndex * BLOCK_SIZE;
                int end = Math.Min(start + BLOCK_SIZE, length);

                int i = start;
                int vectorizableLength = end - start;
                int vectorizedEnd = start + (vectorizableLength / AVX_VECTOR_SIZE) * AVX_VECTOR_SIZE;

                for (; i < vectorizedEnd; i += AVX_VECTOR_SIZE)
                {
                    Vector256<float> aVec = Avx.LoadAlignedVector256(a + i);
                    Vector256<float> positiveMask = Avx.CompareGreaterThan(aVec, _zeroVec);
                    Vector256<float> negativeMask = Avx.CompareLessThan(aVec, _zeroVec);

                    Vector256<float> positivePart = Avx.And(positiveMask, _oneVec);
                    Vector256<float> negativePart = Avx.And(negativeMask, _minusOneVec);
                    Vector256<float> resultVec = Avx.Add(positivePart, negativePart);

                    Avx.StoreAligned(result + i, resultVec);
                }
                for (; i < end; i++)
                {
                    result[i] = MathF.Sign(a[i]);
                }
            });
        }

        public static void Sqrt(float* a, float* result, int length)
        {
            Parallel.For(0, (length + BLOCK_SIZE - 1) / BLOCK_SIZE, _parallelOptions, chunkIndex =>
            {
                int start = chunkIndex * BLOCK_SIZE;
                int end = Math.Min(start + BLOCK_SIZE, length);

                int i = start;
                int vectorizableLength = end - start;
                int vectorizedEnd = start + (vectorizableLength / AVX_VECTOR_SIZE) * AVX_VECTOR_SIZE;

                for (; i < vectorizedEnd; i += AVX_VECTOR_SIZE)
                {
                    Vector256<float> aVec = Avx.LoadAlignedVector256(a + i);
                    Vector256<float> resultVec = Avx.Sqrt(aVec);
                    Avx.StoreAligned(result + i, resultVec);
                }
                for (; i < end; i++)
                {
                    result[i] = MathF.Sqrt(a[i]);
                }
            });
        }

        public static void LogE(float* a, float* result, int length)
        {
            Parallel.For(0, (length + BLOCK_SIZE - 1) / BLOCK_SIZE, _parallelOptions, chunkIndex =>
            {
                int start = chunkIndex * BLOCK_SIZE;
                int end = Math.Min(start + BLOCK_SIZE, length);

                for (int i = start; i < end; i++)
                {
                    result[i] = MathF.Log(a[i]);
                }
            });
        }

        public static void Exp(float* a, float* result, int length)
        {
            Parallel.For(0, (length + BLOCK_SIZE - 1) / BLOCK_SIZE, _parallelOptions, chunkIndex =>
            {
                int start = chunkIndex * BLOCK_SIZE;
                int end = Math.Min(start + BLOCK_SIZE, length);

                for (int i = start; i < end; i++)
                {
                    result[i] = MathF.Exp(a[i]);
                }
            });
        }

        public static unsafe void MatMul(float* a, float* bT, float* result, int aRows, int aCols, int bCols)
        {
            Parallel.For(0, aRows, _parallelOptions, i =>
            {
                float* aRowPtr = a + i * aCols;
                float* resRowPtr = result + i * bCols;

                for (int jInit = 0; jInit < bCols; jInit++)
                {
                    resRowPtr[jInit] = 0;
                }

                for (int kBlock = 0; kBlock < aCols; kBlock += MATRIX_BLOCK_SIZE)
                {
                    int kStart = kBlock;
                    int kEnd = Math.Min(kBlock + MATRIX_BLOCK_SIZE, aCols);

                    for (int j = 0; j < bCols; j++)
                    {
                        float* bColPtr = bT + j * aCols;
                        Vector256<float> sumVec = Vector256<float>.Zero;
                        int k = kStart;

                        for (; k <= kEnd - AVX_VECTOR_SIZE; k += AVX_VECTOR_SIZE)
                        {
                            Vector256<float> aVec = Avx.LoadAlignedVector256(aRowPtr + k);
                            Vector256<float> bVec = Avx.LoadAlignedVector256(bColPtr + k);
                            sumVec = Fma.MultiplyAdd(aVec, bVec, sumVec);
                        }
                        float sum = HorizontalSum(sumVec);
                        for (; k < kEnd; k++)
                        {
                            sum += aRowPtr[k] * bColPtr[k];
                        }

                        resRowPtr[j] += sum;
                    }
                }
            });
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float HorizontalSum(Vector256<float> vec)
        {
            Vector128<float> lower = vec.GetLower();
            Vector128<float> upper = vec.GetUpper();
            Vector128<float> sum128 = Sse42.Add(lower, upper);
            Vector128<float> shuffled = Sse42.Shuffle(sum128, sum128, 0x4E);
            sum128 = Sse42.Add(sum128, shuffled);
            shuffled = Sse42.Shuffle(sum128, sum128, 0xB1);
            sum128 = Sse42.Add(sum128, shuffled);

            return sum128.ToScalar();
        }

        public static void Transpose(float* a, float* result, int rows, int cols)
        {
            int numBlocksI = (rows + MATRIX_BLOCK_SIZE - 1) / MATRIX_BLOCK_SIZE;
            int numBlocksJ = (cols + MATRIX_BLOCK_SIZE - 1) / MATRIX_BLOCK_SIZE;

            Parallel.For(0, numBlocksI * numBlocksJ, _parallelOptions, blockIdx =>
            {
                int blockI = blockIdx / numBlocksJ;
                int blockJ = blockIdx % numBlocksJ;
                int iStart = blockI * MATRIX_BLOCK_SIZE;
                int jStart = blockJ * MATRIX_BLOCK_SIZE;
                int iEnd = Math.Min(iStart + MATRIX_BLOCK_SIZE, rows);
                int jEnd = Math.Min(jStart + MATRIX_BLOCK_SIZE, cols);

                for (int i = iStart; i < iEnd; i++)
                {
                    for (int j = jStart; j < jEnd; j++)
                    {
                        int srcIndex = i * cols + j;
                        int destIndex = j * rows + i;
                        result[destIndex] = a[srcIndex];
                    }
                }
            });
        }
    }
}