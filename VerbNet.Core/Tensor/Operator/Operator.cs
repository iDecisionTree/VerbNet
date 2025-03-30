using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Threading.Tasks;
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
    }
}
