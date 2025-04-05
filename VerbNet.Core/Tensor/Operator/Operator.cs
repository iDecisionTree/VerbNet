using System.Runtime.Intrinsics.X86;
using static System.Reflection.Metadata.BlobBuilder;

namespace VerbNet.Core
{
    public static class Operator
    {
        public static float[] Add(float[] a, float[] b)
        {
            if (Avx.IsSupported)
            {
                return SimdOperator.Add(a, b);
            }
            else
            {
                return ScalarOperator.Add(a, b);
            }
        }

        public static float[] Subtract(float[] a, float[] b)
        {
            if (Avx.IsSupported)
            {
                return SimdOperator.Subtract(a, b);
            }
            else
            {
                return ScalarOperator.Subtract(a, b);
            }
        }

        public static float[] Multiply(float[] a, float[] b)
        {
            if (Avx.IsSupported)
            {
                return SimdOperator.Multiply(a, b);
            }
            else
            {
                return ScalarOperator.Multiply(a, b);
            }
        }

        public static float[] Divide(float[] a, float[] b)
        {
            if (Avx.IsSupported)
            {
                return SimdOperator.Divide(a, b);
            }
            else
            {
                return ScalarOperator.Divide(a, b);
            }
        }

        public static float[] Negate(float[] a)
        {
            if (Avx.IsSupported)
            {
                return SimdOperator.Negate(a);
            }
            else
            {
                return ScalarOperator.Negate(a);
            }
        }

        public static float[] Abs(float[] a)
        {
            if (Avx.IsSupported)
            {
                return SimdOperator.Abs(a);
            }
            else
            {
                return ScalarOperator.Abs(a);
            }
        }

        public static float[] Sign(float[] a)
        {
            if (Avx.IsSupported)
            {
                return SimdOperator.Sign(a);
            }
            else
            {
                return ScalarOperator.Sign(a);
            }
        }

        public static float[] Sqrt(float[] a)
        {
            if (Avx.IsSupported)
            {
                return SimdOperator.Sqrt(a);
            }
            else
            {
                return ScalarOperator.Sqrt(a);
            }
        }

        public static float[] Sin(float[] a)
        {
            return ScalarOperator.Sin(a);
        }

        public static float[] Cos(float[] a)
        {
            return ScalarOperator.Cos(a);
        }

        public static float[] Tan(float[] a)
        {
            return ScalarOperator.Tan(a);
        }

        public static float[] Sinh(float[] a)
        {
            return ScalarOperator.Sinh(a);
        }

        public static float[] Cosh(float[] a)
        {
            return ScalarOperator.Cosh(a);
        }

        public static float[] Tanh(float[] a)
        {
            return ScalarOperator.Tanh(a);
        }

        public static float[] MatMul(float[] a, float[] b, int aRows, int aCols, int bCols)
        {
            if (Avx.IsSupported)
            {
                return SimdOperator.MatMul(a, b, aRows, aCols, bCols);
            }
            else
            {
                return ScalarOperator.MatMul(a, b, aRows, aCols, bCols);
            }
        }

        public static float[] Transpose(float[] a, int rows, int cols)
        {
            return ScalarOperator.Transpose(a, rows, cols);
        }
    }
}
