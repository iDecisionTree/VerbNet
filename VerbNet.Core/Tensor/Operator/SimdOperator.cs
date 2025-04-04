using System.Collections.Concurrent;
using System.Numerics;

namespace VerbNet.Core
{
    public class SimdOperator
    {
        public const int VECTOR_SIZE = 8;

        public static float[] Add(float[] a, float[] b)
        {
            float[] result = new float[a.Length];

            const int chunkSize = 4096;
            Parallel.ForEach(Partitioner.Create(0, result.Length, chunkSize), range =>
            {
                int start = range.Item1;
                int end = range.Item2;

                int simdEnd = start + ((end - start) / VECTOR_SIZE) * VECTOR_SIZE;
                for (int i = start; i < simdEnd; i += VECTOR_SIZE)
                {
                    Vector<float> aVec = new Vector<float>(a, i);
                    Vector<float> bVec = new Vector<float>(b, i);
                    Vector<float> resultVec = Vector.Add(aVec, bVec);
                    resultVec.CopyTo(result, i);
                }
                start = simdEnd;

                for (int i = start; i < end; i++)
                {
                    result[i] = a[i] + b[i];
                }
            });

            return result;
        }

        public static float[] Subtract(float[] a, float[] b)
        {
            float[] result = new float[a.Length];

            const int chunkSize = 4096;
            Parallel.ForEach(Partitioner.Create(0, result.Length, chunkSize), range =>
            {
                int start = range.Item1;
                int end = range.Item2;

                int simdEnd = start + ((end - start) / VECTOR_SIZE) * VECTOR_SIZE;
                for (int i = start; i < simdEnd; i += VECTOR_SIZE)
                {
                    Vector<float> aVec = new Vector<float>(a, i);
                    Vector<float> bVec = new Vector<float>(b, i);
                    Vector<float> resultVec = Vector.Subtract(aVec, bVec);
                    resultVec.CopyTo(result, i);
                }
                start = simdEnd;

                for (int i = start; i < end; i++)
                {
                    result[i] = a[i] - b[i];
                }
            });

            return result;
        }

        public static float[] Multiply(float[] a, float[] b)
        {
            float[] result = new float[a.Length];

            const int chunkSize = 4096;
            Parallel.ForEach(Partitioner.Create(0, result.Length, chunkSize), range =>
            {
                int start = range.Item1;
                int end = range.Item2;

                int simdEnd = start + ((end - start) / VECTOR_SIZE) * VECTOR_SIZE;
                for (int i = start; i < simdEnd; i += VECTOR_SIZE)
                {
                    Vector<float> aVec = new Vector<float>(a, i);
                    Vector<float> bVec = new Vector<float>(b, i);
                    Vector<float> resultVec = Vector.Multiply(aVec, bVec);
                    resultVec.CopyTo(result, i);
                }
                start = simdEnd;

                for (int i = start; i < end; i++)
                {
                    result[i] = a[i] * b[i];
                }
            });

            return result;
        }

        public static float[] Divide(float[] a, float[] b)
        {
            float[] result = new float[a.Length];

            const int chunkSize = 4096;
            Parallel.ForEach(Partitioner.Create(0, result.Length, chunkSize), range =>
            {
                int start = range.Item1;
                int end = range.Item2;

                int simdEnd = start + ((end - start) / VECTOR_SIZE) * VECTOR_SIZE;
                for (int i = start; i < simdEnd; i += VECTOR_SIZE)
                {
                    Vector<float> aVec = new Vector<float>(a, i);
                    Vector<float> bVec = new Vector<float>(b, i);
                    Vector<float> resultVec = Vector.Divide(aVec, bVec);
                    resultVec.CopyTo(result, i);
                }
                start = simdEnd;

                for (int i = start; i < end; i++)
                {
                    result[i] = a[i] / b[i];
                }
            });

            return result;
        }

        public static float[] Negate(float[] a)
        {
            float[] result = new float[a.Length];

            const int chunkSize = 4096;
            Parallel.ForEach(Partitioner.Create(0, result.Length, chunkSize), range =>
            {
                int start = range.Item1;
                int end = range.Item2;

                int simdEnd = start + ((end - start) / VECTOR_SIZE) * VECTOR_SIZE;
                for (int i = start; i < simdEnd; i += VECTOR_SIZE)
                {
                    Vector<float> aVec = new Vector<float>(a, i);
                    Vector<float> resultVec = Vector.Negate(aVec);
                    resultVec.CopyTo(result, i);
                }
                start = simdEnd;

                for (int i = start; i < end; i++)
                {
                    result[i] = -a[i];
                }
            });

            return result;
        }

        public static float[] MatMul(float[] a, float[] b, int aRows, int aCols, int bCols)
        {
            const int blockSize = 64;
            float[] result = new float[aRows * bCols];
            float[] bTransposed = TransposeMatrix(b, aCols, bCols);

            Parallel.ForEach(Partitioner.Create(0, aRows, blockSize), rowRange =>
            {
                var simdAccumulators = new Vector<float>[blockSize * blockSize];
                int rowStart = rowRange.Item1;
                int rowEnd = Math.Min(rowRange.Item2, aRows);

                for (int kk = 0; kk < aCols; kk += blockSize)
                {
                    int kEnd = Math.Min(kk + blockSize, aCols);

                    for (int jj = 0; jj < bCols; jj += blockSize)
                    {
                        int jEnd = Math.Min(jj + blockSize, bCols);

                        for (int i = rowStart; i < rowEnd; i++)
                        {
                            for (int j = jj; j < jEnd; j += Vector<float>.Count)
                            {
                                int remaining = Math.Min(Vector<float>.Count, jEnd - j);
                                Vector<float> sumVec = Vector<float>.Zero;

                                for (int k = kk; k < kEnd; k++)
                                {
                                    Vector<float> aVec = new Vector<float>(a[i * aCols + k]);
                                    Vector<float> bVec = new Vector<float>(bTransposed, j * aCols + k);
                                    sumVec += aVec * bVec;
                                }

                                if (remaining == Vector<float>.Count)
                                {
                                    sumVec.CopyTo(result, i * bCols + j);
                                }
                                else
                                {
                                    for (int n = 0; n < remaining; n++)
                                    {
                                        result[i * bCols + j + n] += sumVec[n];
                                    }
                                }
                            }
                        }
                    }
                }
            });

            return result;
        }

        private static float[] TransposeMatrix(float[] matrix, int rows, int cols)
        {
            float[] transposed = new float[matrix.Length];
            Parallel.For(0, rows, i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    transposed[j * rows + i] = matrix[i * cols + j];
                }
            });

            return transposed;
        }
    }
}
