namespace VerbNet.Core
{
    public static class ScalarOperator
    {
        public static float[] Add(float[] a, float[] b)
        {
            float[] result = new float[a.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = a[i] + b[i];
            }

            return result;
        }

        public static float[] Subtract(float[] a, float[] b)
        {
            float[] result = new float[a.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = a[i] - b[i];
            }

            return result;
        }

        public static float[] Multiply(float[] a, float[] b)
        {
            float[] result = new float[a.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = a[i] * b[i];
            }

            return result;
        }

        public static float[] Divide(float[] a, float[] b)
        {
            float[] result = new float[a.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = a[i] / b[i];
            }

            return result;
        }

        public static float[] Negate(float[] a)
        {
            float[] result = new float[a.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = -a[i];
            }

            return result;
        }

        public static float[] Abs(float[] a)
        {
            float[] result = new float[a.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = Math.Abs(a[i]);
            }

            return result;
        }

        public static float[] Sign(float[] a)
        {
            float[] result = new float[a.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = Math.Sign(a[i]);
            }

            return result;
        }

        public static float[] Sqrt(float[] a)
        {
            float[] result = new float[a.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = MathF.Sqrt(a[i]);
            }

            return result;
        }

        public static float[] LogE(float[] a)
        {
            float[] result = new float[a.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = MathF.Log(a[i]);
            }

            return result;
        }

        public static float[] Exp(float[] a)
        {
            float[] result = new float[a.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = MathF.Exp(a[i]);
            }

            return result;
        }

        public static float[] Sin(float[] a)
        {
            float[] result = new float[a.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = MathF.Sin(a[i]);
            }

            return result;
        }

        public static float[] Cos(float[] a)
        {
            float[] result = new float[a.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = MathF.Cos(a[i]);
            }

            return result;
        }

        public static float[] Tan(float[] a)
        {
            float[] result = new float[a.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = MathF.Tan(a[i]);
            }

            return result;
        }

        public static float[] Sinh(float[] a)
        {
            float[] result = new float[a.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = MathF.Sinh(a[i]);
            }

            return result;
        }

        public static float[] Cosh(float[] a)
        {
            float[] result = new float[a.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = MathF.Cosh(a[i]);
            }

            return result;
        }

        public static float[] Tanh(float[] a)
        {
            float[] result = new float[a.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = MathF.Tanh(a[i]);
            }

            return result;
        }

        public static float[] MatMul(float[] a, float[] b, int aRows, int aCols, int bCols)
        {
            float[] result = new float[aRows * bCols];
            Parallel.For(0, aRows, i =>
            {
                for (int j = 0; j < bCols; j++)
                {
                    float sum = 0f;
                    for (int k = 0; k < aCols; k++)
                    {
                        sum += a[i * aCols + k] * b[k * bCols + j];
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
