using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using System.Runtime.CompilerServices;

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
            const int chunkSize = 4096;
            Vector256<float> zeroVec = Vector256<float>.Zero;

            Parallel.For(0, (length + chunkSize - 1) / chunkSize, _parallelOptions, chunkIndex =>
            {
                int start = chunkIndex * chunkSize;
                int end = Math.Min(start + chunkSize, length);

                int i = start;
                int vectorizableLength = end - start;
                int vectorizedEnd = start + (vectorizableLength / AVX_VECTOR_SIZE) * AVX_VECTOR_SIZE;

                for (; i < vectorizedEnd; i += AVX_VECTOR_SIZE)
                {
                    Vector256<float> aVec = Avx.LoadAlignedVector256(a + i);
                    Vector256<float> resultVec = Avx.Subtract(zeroVec, aVec);
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
            Vector256<float> zeroVec = Vector256<float>.Zero;

            Parallel.For(0, (length + chunkSize - 1) / chunkSize, _parallelOptions, chunkIndex =>
            {
                int start = chunkIndex * chunkSize;
                int end = Math.Min(start + chunkSize, length);

                int i = start;
                int vectorizableLength = end - start;
                int vectorizedEnd = start + (vectorizableLength / AVX_VECTOR_SIZE) * AVX_VECTOR_SIZE;

                for (; i < vectorizedEnd; i += AVX_VECTOR_SIZE)
                {
                    Vector256<float> aVec = Avx.LoadAlignedVector256(a + i);
                    Vector256<float> mask = Avx.CompareLessThan(aVec, zeroVec);
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
            const int chunkSize = 4096;
            Vector256<float> zeroVec = Vector256<float>.Zero;
            Vector256<float> oneVec = Vector256.Create(1.0f);
            Vector256<float> minusOneVec = Vector256.Create(-1.0f);

            Parallel.For(0, (length + chunkSize - 1) / chunkSize, _parallelOptions, chunkIndex =>
            {
                int start = chunkIndex * chunkSize;
                int end = Math.Min(start + chunkSize, length);

                int i = start;
                int vectorizableLength = end - start;
                int vectorizedEnd = start + (vectorizableLength / AVX_VECTOR_SIZE) * AVX_VECTOR_SIZE;

                for (; i < vectorizedEnd; i += AVX_VECTOR_SIZE)
                {
                    Vector256<float> aVec = Avx.LoadAlignedVector256(a + i);
                    Vector256<float> positiveMask = Avx.CompareGreaterThan(aVec, zeroVec);
                    Vector256<float> negativeMask = Avx.CompareLessThan(aVec, zeroVec);

                    Vector256<float> positivePart = Avx.And(positiveMask, oneVec);
                    Vector256<float> negativePart = Avx.And(negativeMask, minusOneVec);
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
                float* aRowPtr = a + i * aCols;

                for (int j = 0; j < bCols; j++)
                {
                    float* bColPtr = bT + j * aCols;
                    Vector256<float> sumVec = Vector256<float>.Zero;

                    int k = 0;
                    for (; k <= aCols - AVX_VECTOR_SIZE; k += AVX_VECTOR_SIZE)
                    {
                        var aVec = Avx.LoadAlignedVector256(aRowPtr + k);
                        var bVec = Avx.LoadAlignedVector256(bColPtr + k);
                        sumVec = Avx.Add(sumVec, Avx.Multiply(aVec, bVec));
                    }
                    float sum = HorizontalSum(sumVec);

                    for (; k < aCols; k++)
                    {
                        sum += aRowPtr[k] * bColPtr[k];
                    }

                    result[i * bCols + j] = sum;
                }
            });
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float HorizontalSum(Vector256<float> vec)
        {
            Vector128<float> lower = vec.GetLower();
            Vector128<float> upper = vec.GetUpper();
            Vector128<float> sum128 = Sse.Add(lower, upper);
            Vector128<float> shuffled = Sse.Shuffle(sum128, sum128, 0x4E);
            sum128 = Sse.Add(sum128, shuffled);
            shuffled = Sse.Shuffle(sum128, sum128, 0xB1);
            sum128 = Sse.Add(sum128, shuffled);

            return sum128.ToScalar();
        }
    }
}