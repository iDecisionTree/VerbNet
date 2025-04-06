using System.Runtime.Intrinsics.X86;

namespace VerbNet.Core
{
    public static unsafe class Operator
    {
        public static AlignedArray<float> Add(AlignedArray<float> a, AlignedArray<float> b)
        {
            AlignedArray<float> result = new AlignedArray<float>(a.Length, a.Alignment);
            if (Avx.IsSupported)
            {
                SimdOperator.Add(a.Ptr, b.Ptr, result.Ptr, a.Length);
            }
            else
            {
                ScalarOperator.Add(a.Ptr, b.Ptr, result.Ptr, a.Length);
            }

            return result;
        }

        public static AlignedArray<float> Subtract(AlignedArray<float> a, AlignedArray<float> b)
        {
            AlignedArray<float> result = new AlignedArray<float>(a.Length, a.Alignment);
            if (Avx.IsSupported)
            {
                SimdOperator.Subtract(a.Ptr, b.Ptr, result.Ptr, a.Length);
            }
            else
            {
                ScalarOperator.Subtract(a.Ptr, b.Ptr, result.Ptr, a.Length);
            }

            return result;
        }

        public static AlignedArray<float> Multiply(AlignedArray<float> a, AlignedArray<float> b)
        {
            AlignedArray<float> result = new AlignedArray<float>(a.Length, a.Alignment);
            if (Avx.IsSupported)
            {
                SimdOperator.Multiply(a.Ptr, b.Ptr, result.Ptr, a.Length);
            }
            else
            {
                ScalarOperator.Multiply(a.Ptr, b.Ptr, result.Ptr, a.Length);
            }

            return result;
        }

        public static AlignedArray<float> Divide(AlignedArray<float> a, AlignedArray<float> b)
        {
            AlignedArray<float> result = new AlignedArray<float>(a.Length, a.Alignment);
            if (Avx.IsSupported)
            {
                SimdOperator.Divide(a.Ptr, b.Ptr, result.Ptr, a.Length);
            }
            else
            {
                ScalarOperator.Divide(a.Ptr, b.Ptr, result.Ptr, a.Length);
            }

            return result;
        }

        public static AlignedArray<float> Negate(AlignedArray<float> a)
        {
            AlignedArray<float> result = new AlignedArray<float>(a.Length, a.Alignment);
            if (Avx.IsSupported)
            {
                SimdOperator.Negate(a.Ptr, result.Ptr, a.Length);
            }
            else
            {
                ScalarOperator.Negate(a.Ptr, result.Ptr, a.Length);
            }

            return result;
        }

        public static AlignedArray<float> Abs(AlignedArray<float> a)
        {
            AlignedArray<float> result = new AlignedArray<float>(a.Length, a.Alignment);
            if (Avx.IsSupported)
            {
                SimdOperator.Abs(a.Ptr, result.Ptr, a.Length);
            }
            else
            {
                ScalarOperator.Abs(a.Ptr, result.Ptr, a.Length);
            }

            return result;
        }

        public static AlignedArray<float> Sign(AlignedArray<float> a)
        {
            AlignedArray<float> result = new AlignedArray<float>(a.Length, a.Alignment);
            if (Avx.IsSupported)
            {
                SimdOperator.Sign(a.Ptr, result.Ptr, a.Length);
            }
            else
            {
                ScalarOperator.Sign(a.Ptr, result.Ptr, a.Length);
            }

            return result;
        }

        public static AlignedArray<float> Sqrt(AlignedArray<float> a)
        {
            AlignedArray<float> result = new AlignedArray<float>(a.Length, a.Alignment);
            if (Avx.IsSupported)
            {
                SimdOperator.Sqrt(a.Ptr, result.Ptr, a.Length);
            }
            else
            {
                ScalarOperator.Sqrt(a.Ptr, result.Ptr, a.Length);
            }

            return result;
        }

        public static AlignedArray<float> LogE(AlignedArray<float> a)
        {
            AlignedArray<float> result = new AlignedArray<float>(a.Length, a.Alignment);
            if (Avx.IsSupported)
            {
                SimdOperator.LogE(a.Ptr, result.Ptr, a.Length);
            }
            else
            {
                ScalarOperator.LogE(a.Ptr, result.Ptr, a.Length);
            }

            return result;
        }

        public static AlignedArray<float> Exp(AlignedArray<float> a)
        {
            AlignedArray<float> result = new AlignedArray<float>(a.Length, a.Alignment);
            if (Avx.IsSupported)
            {
                SimdOperator.Exp(a.Ptr, result.Ptr, a.Length);
            }
            else
            {
                ScalarOperator.Exp(a.Ptr, result.Ptr, a.Length);
            }

            return result;
        }

        public static AlignedArray<float> Power(AlignedArray<float> a, float exponent)
        {
            AlignedArray<float> result = new AlignedArray<float>(a.Length, a.Alignment);
            ScalarOperator.Power(a.Ptr, exponent, result.Ptr, a.Length);

            return result;
        }

        public static AlignedArray<float> Sin(AlignedArray<float> a)
        {
            AlignedArray<float> result = new AlignedArray<float>(a.Length, a.Alignment);
            ScalarOperator.Sin(a.Ptr, result.Ptr, a.Length);

            return result;
        }

        public static AlignedArray<float> Cos(AlignedArray<float> a)
        {
            AlignedArray<float> result = new AlignedArray<float>(a.Length, a.Alignment);
            ScalarOperator.Cos(a.Ptr, result.Ptr, a.Length);

            return result;
        }

        public static AlignedArray<float> Tan(AlignedArray<float> a)
        {
            AlignedArray<float> result = new AlignedArray<float>(a.Length, a.Alignment);
            ScalarOperator.Tan(a.Ptr, result.Ptr, a.Length);

            return result;
        }

        public static AlignedArray<float> Sinh(AlignedArray<float> a)
        {
            AlignedArray<float> result = new AlignedArray<float>(a.Length, a.Alignment);
            ScalarOperator.Sinh(a.Ptr, result.Ptr, a.Length);

            return result;
        }

        public static AlignedArray<float> Cosh(AlignedArray<float> a)
        {
            AlignedArray<float> result = new AlignedArray<float>(a.Length, a.Alignment);
            ScalarOperator.Cosh(a.Ptr, result.Ptr, a.Length);

            return result;
        }

        public static AlignedArray<float> Tanh(AlignedArray<float> a)
        {
            AlignedArray<float> result = new AlignedArray<float>(a.Length, a.Alignment);
            ScalarOperator.Tanh(a.Ptr, result.Ptr, a.Length);

            return result;
        }

        public static AlignedArray<float> MatMul(AlignedArray<float> a, AlignedArray<float> b, int aRows, int aCols, int bRows, int bCols)
        {
            AlignedArray<float> result = new AlignedArray<float>(aRows * bCols, a.Alignment);
            if (Avx.IsSupported)
            {
                AlignedArray<float> bTransposed = Transpose(b, bRows, bCols);
                SimdOperator.MatMul(a.Ptr, bTransposed.Ptr, result.Ptr, aRows, aCols, bCols);
                bTransposed.Dispose();
            }
            else
            {
                ScalarOperator.MatMul(a.Ptr, b.Ptr, result.Ptr, aRows, aCols, bCols);
            }

            return result;
        }

        public static AlignedArray<float> Transpose(AlignedArray<float> a, int rows, int cols)
        {
            AlignedArray<float> result = new AlignedArray<float>(rows * cols, a.Alignment);
            if (Avx.IsSupported)
            {
                SimdOperator.Transpose(a.Ptr, result.Ptr, rows, cols);
            }
            else
            {
                ScalarOperator.Transpose(a.Ptr, result.Ptr, rows, cols);
            }
            return result;
        }
    }
}