using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace VerbNet.Core
{
    public static class Operator
    {
        public static float[] Add(float[] a, float[] b)
        {
            return ScalarOperator.Add(a, b);
        }

        public static float[] Subtract(float[] a, float[] b)
        {
            return ScalarOperator.Subtract(a, b);
        }

        public static float[] Multiply(float[] a, float[] b)
        {
            return ScalarOperator.Multiply(a, b);
        }

        public static float[] Divide(float[] a, float[] b)
        {
            return ScalarOperator.Divide(a, b);
        }

        public static float[] Negate(float[] a)
        {
            return ScalarOperator.Negate(a);
        }
    }
}
