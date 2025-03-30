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
    }
}
