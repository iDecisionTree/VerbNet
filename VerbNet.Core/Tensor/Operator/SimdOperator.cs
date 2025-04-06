using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;

namespace VerbNet.Core
{
    public static unsafe class SimdOperator
    {
        public const int AVX_VECTOR_SIZE = 8;

        private static int _maxDegreeOfParallelism => Environment.ProcessorCount;
        private static ParallelOptions _parallelOptions = new ParallelOptions
        {
            MaxDegreeOfParallelism = _maxDegreeOfParallelism
        };

        public static void Add(float* a, float* b, float* result, int length)
        {
            const int chunkSize = 4096;

            Parallel.For(0, (length + chunkSize - 1) / chunkSize, _parallelOptions, chunkIndex =>
            {
                int start = chunkIndex * chunkSize;
                int end = Math.Min(start + chunkSize, length);

                int i = start;
                int vectorizableLength = end - start;
                int vectorizedEnd = start + (vectorizableLength / AVX_VECTOR_SIZE) * AVX_VECTOR_SIZE;

                for (; i < vectorizedEnd; i += AVX_VECTOR_SIZE)
                {
                    Vector128<float> aVec = Avx.LoadAlignedVector128(a + i);
                    Vector128<float> bVec = Avx.LoadAlignedVector128(b + i);
                    Vector128<float> resultVec = Avx.Add(aVec, bVec);
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
            const int chunkSize = 4096;

            Parallel.For(0, (length + chunkSize - 1) / chunkSize, _parallelOptions, chunkIndex =>
            {
                int start = chunkIndex * chunkSize;
                int end = Math.Min(start + chunkSize, length);

                int i = start;
                int vectorizableLength = end - start;
                int vectorizedEnd = start + (vectorizableLength / AVX_VECTOR_SIZE) * AVX_VECTOR_SIZE;

                for (; i < vectorizedEnd; i += AVX_VECTOR_SIZE)
                {
                    Vector128<float> aVec = Avx.LoadAlignedVector128(a + i);
                    Vector128<float> bVec = Avx.LoadAlignedVector128(b + i);
                    Vector128<float> resultVec = Avx.Subtract(aVec, bVec);
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
            const int chunkSize = 4096;

            Parallel.For(0, (length + chunkSize - 1) / chunkSize, _parallelOptions, chunkIndex =>
            {
                int start = chunkIndex * chunkSize;
                int end = Math.Min(start + chunkSize, length);

                int i = start;
                int vectorizableLength = end - start;
                int vectorizedEnd = start + (vectorizableLength / AVX_VECTOR_SIZE) * AVX_VECTOR_SIZE;

                for (; i < vectorizedEnd; i += AVX_VECTOR_SIZE)
                {
                    Vector128<float> aVec = Avx.LoadAlignedVector128(a + i);
                    Vector128<float> bVec = Avx.LoadAlignedVector128(b + i);
                    Vector128<float> resultVec = Avx.Multiply(aVec, bVec);
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
            const int chunkSize = 4096;

            Parallel.For(0, (length + chunkSize - 1) / chunkSize, _parallelOptions, chunkIndex =>
            {
                int start = chunkIndex * chunkSize;
                int end = Math.Min(start + chunkSize, length);

                int i = start;
                int vectorizableLength = end - start;
                int vectorizedEnd = start + (vectorizableLength / AVX_VECTOR_SIZE) * AVX_VECTOR_SIZE;

                for (; i < vectorizedEnd; i += AVX_VECTOR_SIZE)
                {
                    Vector128<float> aVec = Avx.LoadAlignedVector128(a + i);
                    Vector128<float> bVec = Avx.LoadAlignedVector128(b + i);
                    Vector128<float> resultVec = Avx.Divide(aVec, bVec);
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
            const int chunkSize = 4096;
            Vector128<float> zeroVec = Vector128<float>.Zero;

            Parallel.For(0, (length + chunkSize - 1) / chunkSize, _parallelOptions, chunkIndex =>
            {
                int start = chunkIndex * chunkSize;
                int end = Math.Min(start + chunkSize, length);

                int i = start;
                int vectorizableLength = end - start;
                int vectorizedEnd = start + (vectorizableLength / AVX_VECTOR_SIZE) * AVX_VECTOR_SIZE;

                for (; i < vectorizedEnd; i += AVX_VECTOR_SIZE)
                {
                    Vector128<float> aVec = Avx.LoadAlignedVector128(a + i);
                    Vector128<float> resultVec = Avx.Subtract(zeroVec, aVec);
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
            const int chunkSize = 4096;
            Vector128<float> zeroVec = Vector128<float>.Zero;

            Parallel.For(0, (length + chunkSize - 1) / chunkSize, _parallelOptions, chunkIndex =>
            {
                int start = chunkIndex * chunkSize;
                int end = Math.Min(start + chunkSize, length);

                int i = start;
                int vectorizableLength = end - start;
                int vectorizedEnd = start + (vectorizableLength / AVX_VECTOR_SIZE) * AVX_VECTOR_SIZE;

                for (; i < vectorizedEnd; i += AVX_VECTOR_SIZE)
                {
                    Vector128<float> aVec = Avx.LoadAlignedVector128(a + i);
                    Vector128<float> mask = Avx.CompareLessThan(aVec, zeroVec);
                    Vector128<float> resultVec = Avx.Xor(aVec, mask);
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
            const int chunkSize = 4096;
            Vector128<float> zeroVec = Vector128<float>.Zero;
            Vector128<float> oneVec = Vector128.Create(1.0f);
            Vector128<float> minusOneVec = Vector128.Create(-1.0f);

            Parallel.For(0, (length + chunkSize - 1) / chunkSize, _parallelOptions, chunkIndex =>
            {
                int start = chunkIndex * chunkSize;
                int end = Math.Min(start + chunkSize, length);

                int i = start;
                int vectorizableLength = end - start;
                int vectorizedEnd = start + (vectorizableLength / AVX_VECTOR_SIZE) * AVX_VECTOR_SIZE;

                for (; i < vectorizedEnd; i += AVX_VECTOR_SIZE)
                {
                    Vector128<float> aVec = Avx.LoadAlignedVector128(a + i);
                    Vector128<float> positiveMask = Avx.CompareGreaterThan(aVec, zeroVec);
                    Vector128<float> negativeMask = Avx.CompareLessThan(aVec, zeroVec);

                    Vector128<float> positivePart = Avx.And(positiveMask, oneVec);
                    Vector128<float> negativePart = Avx.And(negativeMask, minusOneVec);
                    Vector128<float> resultVec = Avx.Add(positivePart, negativePart);

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
            const int chunkSize = 4096;

            Parallel.For(0, (length + chunkSize - 1) / chunkSize, _parallelOptions, chunkIndex =>
            {
                int start = chunkIndex * chunkSize;
                int end = Math.Min(start + chunkSize, length);

                int i = start;
                int vectorizableLength = end - start;
                int vectorizedEnd = start + (vectorizableLength / AVX_VECTOR_SIZE) * AVX_VECTOR_SIZE;

                for (; i < vectorizedEnd; i += AVX_VECTOR_SIZE)
                {
                    Vector128<float> aVec = Avx.LoadAlignedVector128(a + i);
                    Vector128<float> resultVec = Avx.Sqrt(aVec);
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
            const int chunkSize = 4096;

            Parallel.For(0, (length + chunkSize - 1) / chunkSize, _parallelOptions, chunkIndex =>
            {
                int start = chunkIndex * chunkSize;
                int end = Math.Min(start + chunkSize, length);

                for (int i = start; i < end; i++)
                {
                    result[i] = MathF.Log(a[i]);
                }
            });
        }

        public static void Exp(float* a, float* result, int length)
        {
            const int chunkSize = 4096;

            Parallel.For(0, (length + chunkSize - 1) / chunkSize, _parallelOptions, chunkIndex =>
            {
                int start = chunkIndex * chunkSize;
                int end = Math.Min(start + chunkSize, length);

                for (int i = start; i < end; i++)
                {
                    result[i] = MathF.Exp(a[i]);
                }
            });
        }

        public static void MatMul(float* a, float* bT, float* result, int aRows, int aCols, int bCols)
        {
            Parallel.For(0, aRows, i =>
            {
                for (int j = 0; j < bCols; j++)
                {
                    float sum = 0;
                    int k = 0;
                    for (; k <= aCols - AVX_VECTOR_SIZE; k += AVX_VECTOR_SIZE)
                    {
                        Vector128<float> aVec = Avx.LoadAlignedVector128(a + i * aCols + k);
                        Vector128<float> bVec = Avx.LoadAlignedVector128(bT + j * aCols + k);
                        Vector128<float> mul = Avx.Multiply(aVec, bVec);

                        Vector128<float> sum128 = Sse3.HorizontalAdd(mul, mul);
                        sum128 = Sse3.HorizontalAdd(sum128, sum128);
                        sum += sum128.ToScalar();
                    }
                    for (; k < aCols; k++)
                    {
                        sum += a[i * aCols + k] * bT[j * aCols + k];
                    }
                    result[i * bCols + j] = sum;
                }
            });
        }

        public static void Transpose(float* a, float* result, int rows, int cols)
        {
            Parallel.For(0, rows, i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    int sourceIndex = i * cols + j;
                    int destIndex = j * rows + i;
                    result[destIndex] = a[sourceIndex];
                }
            });
        }
    }
}