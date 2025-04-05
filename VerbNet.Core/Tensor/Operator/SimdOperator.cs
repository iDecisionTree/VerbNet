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

        public static float[] Abs(float[] a)
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
                    Vector<float> resultVec = Vector.Abs(aVec);
                    resultVec.CopyTo(result, i);
                }
                start = simdEnd;

                for (int i = start; i < end; i++)
                {
                    result[i] = Math.Abs(a[i]);
                }
            });

            return result;
        }

        public static float[] Sign(float[] a)
        {
            float[] result = new float[a.Length];

            const int chunkSize = 4096;
            Vector<float> oneVector = new Vector<float>(1f);
            Vector<float> minusOneVector = new Vector<float>(-1f);
            Vector<float> zeroVector = Vector<float>.Zero;

            Parallel.ForEach(Partitioner.Create(0, result.Length, chunkSize), range =>
            {
                int start = range.Item1;
                int end = range.Item2;

                int simdEnd = start + ((end - start) / VECTOR_SIZE) * VECTOR_SIZE;
                for (int i = start; i < simdEnd; i += VECTOR_SIZE)
                {
                    Vector<float> inputVec = new Vector<float>(a, i);

                    Vector<int> positiveMask = Vector.GreaterThan(inputVec, zeroVector);
                    Vector<int> negativeMask = Vector.LessThan(inputVec, zeroVector);

                    Vector<float> positivePart = Vector.ConditionalSelect(positiveMask, oneVector, zeroVector);
                    Vector<float> negativePart = Vector.ConditionalSelect(negativeMask, minusOneVector, zeroVector);
                    Vector<float> resultVec = positivePart + negativePart;

                    resultVec.CopyTo(result, i);
                }

                for (int i = simdEnd; i < end; i++)
                {
                    result[i] = Math.Sign(a[i]);
                }
            });

            return result;
        }

        public static float[] Sqrt(float[] a)
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
                    Vector<float> resultVec = Vector.SquareRoot(aVec);
                    resultVec.CopyTo(result, i);
                }
                start = simdEnd;

                for (int i = start; i < end; i++)
                {
                    result[i] = MathF.Sqrt(a[i]);
                }
            });

            return result;
        }

        public static float[] LogE(float[] a)
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
                    Vector<float> resultVec = Vector.Log(aVec);
                    resultVec.CopyTo(result, i);
                }
                start = simdEnd;

                for (int i = start; i < end; i++)
                {
                    result[i] = MathF.Log(a[i]);
                }
            });

            return result;
        }

        public static float[] MatMul(float[] a, float[] b, int aRows, int aCols, int bCols)
        {
            float[] result = new float[aRows * bCols];
            float[] bTransposed = Transpose(b, aCols, bCols);

            Parallel.For(0, aRows, i =>
            {
                for (int j = 0; j < bCols; j++)
                {
                    float sum = 0;
                    int k = 0;
                    for (; k <= aCols - Vector<float>.Count; k += Vector<float>.Count)
                    {
                        var aVec = new Vector<float>(a, i * aCols + k);
                        var bVec = new Vector<float>(bTransposed, j * aCols + k);
                        sum += Vector.Dot(aVec, bVec);
                    }
                    for (; k < aCols; k++)
                    {
                        sum += a[i * aCols + k] * bTransposed[j * aCols + k];
                    }
                    result[i * bCols + j] = sum;
                }
            });

            return result;
        }

        public static float[] Transpose(float[] a, int rows, int cols)
        {
            float[] result = new float[rows * cols];
            Parallel.For(0, rows, i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    int sourceIndex = i * cols + j;
                    int destIndex = j * rows + i;
                    result[destIndex] = a[sourceIndex];
                }
            });

            return result;
        }
    }
}
