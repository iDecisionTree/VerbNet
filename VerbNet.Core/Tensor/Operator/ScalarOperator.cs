using System.Reflection.Metadata.Ecma335;

namespace VerbNet.Core
{
    public static unsafe class ScalarOperator
    {
        public static void Add(float* a, float* b, float* result, int length)
        {
            for (int i = 0; i < length; i++)
            {
                result[i] = a[i] + b[i];
            }
        }

        public static void Add(float* a, float b, float* result, int length)
        {
            for (int i = 0; i < length; i++)
            {
                result[i] = a[i] + b;
            }
        }

        public static void Subtract(float* a, float* b, float* result, int length)
        {
            for (int i = 0; i < length; i++)
            {
                result[i] = a[i] - b[i];
            }
        }

        public static void Substract(float* a, float b, float* result, int length)
        {
            for (int i = 0; i < length; i++)
            {
                result[i] = a[i] - b;
            }
        }

        public static void Substract(float a, float* b, float* result, int length)
        {
            for (int i = 0; i < length; i++)
            {
                result[i] = a - b[i];
            }
        }

        public static void Multiply(float* a, float* b, float* result, int length)
        {
            for (int i = 0; i < length; i++)
            {
                result[i] = a[i] * b[i];
            }
        }

        public static void Multiply(float* a, float b, float* result, int length)
        {
            for (int i = 0; i < length; i++)
            {
                result[i] = a[i] * b;
            }
        }

        public static void Divide(float* a, float* b, float* result, int length)
        {
            for (int i = 0; i < length; i++)
            {
                result[i] = a[i] / b[i];
            }
        }

        public static void Divide(float* a, float b, float* result, int length)
        {
            for (int i = 0; i < length; i++)
            {
                result[i] = a[i] / b;
            }
        }

        public static void Divide(float a, float* b, float* result, int length)
        {
            for (int i = 0; i < length; i++)
            {
                result[i] = a / b[i];
            }
        }

        public static void Negate(float* a, float* result, int length)
        {
            for (int i = 0; i < length; i++)
            {
                result[i] = -a[i];
            }
        }

        public static void Abs(float* a, float* result, int length)
        {
            for (int i = 0; i < length; i++)
            {
                result[i] = Math.Abs(a[i]);
            }
        }

        public static void Sign(float* a, float* result, int length)
        {
            for (int i = 0; i < length; i++)
            {
                result[i] = Math.Sign(a[i]);
            }
        }

        public static void Sqrt(float* a, float* result, int length)
        {
            for (int i = 0; i < length; i++)
            {
                result[i] = MathF.Sqrt(a[i]);
            }
        }

        public static void LogE(float* a, float* result, int length)
        {
            for (int i = 0; i < length; i++)
            {
                result[i] = MathF.Log(a[i]);
            }
        }

        public static void Exp(float* a, float* result, int length)
        {
            for (int i = 0; i < length; i++)
            {
                result[i] = MathF.Exp(a[i]);
            }
        }

        public static void Power(float* a, float exponent, float* result, int length)
        {
            for (int i = 0; i < length; i++)
            {
                result[i] = MathF.Pow(a[i], exponent);
            }
        }

        public static void Sin(float* a, float* result, int length)
        {
            for (int i = 0; i < length; i++)
            {
                result[i] = MathF.Sin(a[i]);
            }
        }

        public static void Cos(float* a, float* result, int length)
        {
            for (int i = 0; i < length; i++)
            {
                result[i] = MathF.Cos(a[i]);
            }
        }

        public static void Tan(float* a, float* result, int length)
        {
            for (int i = 0; i < length; i++)
            {
                result[i] = MathF.Tan(a[i]);
            }
        }

        public static void Sinh(float* a, float* result, int length)
        {
            for (int i = 0; i < length; i++)
            {
                result[i] = MathF.Sinh(a[i]);
            }
        }

        public static void Cosh(float* a, float* result, int length)
        {
            for (int i = 0; i < length; i++)
            {
                result[i] = MathF.Cosh(a[i]);
            }
        }

        public static void Tanh(float* a, float* result, int length)
        {
            for (int i = 0; i < length; i++)
            {
                result[i] = MathF.Tanh(a[i]);
            }
        }

        public static void MatMul(float* a, float* b, float* result, int aRows, int aCols, int bCols)
        {
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