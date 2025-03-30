using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
    }
}
