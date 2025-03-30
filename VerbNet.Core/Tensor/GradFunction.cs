using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace VerbNet.Core
{
    public static class GradFunction
    {
        public static (Tensor, Tensor) AddGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf)
        {
            return (gradient, gradient);
        }

        public static (Tensor, Tensor) SubGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf)
        {
            return (gradient, TensorOperator.Negate(gradient, false));
        }

        public static (Tensor, Tensor) MulGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf)
        {
            return (TensorOperator.Multiply(gradient, rightLeaf, false), TensorOperator.Multiply(gradient, leftLeaf, false));
        }

        public static (Tensor, Tensor) DivGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf)
        {
            return (TensorOperator.Divide(gradient, rightLeaf, false), TensorOperator.Divide(TensorOperator.Multiply(TensorOperator.Negate(leftLeaf, false), gradient, false), TensorOperator.Multiply(rightLeaf, rightLeaf, false), false));
        }

        public static (Tensor, Tensor) NegGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf)
        {
            return (TensorOperator.Negate(gradient, false), null);
        }

        public static (Tensor, Tensor) MatMulGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf)
        {
            Tensor bTransposed = TensorOperator.Transpose(rightLeaf, false);
            Tensor dA = TensorOperator.MatMul(gradient, bTransposed, false);

            Tensor aTransposed = TensorOperator.Transpose(leftLeaf, false);
            Tensor dB = TensorOperator.MatMul(aTransposed, gradient, false);

            return (dA, dB);
        }

        public static (Tensor, Tensor) TransposeGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf)
        {
            Tensor gradTransposed = TensorOperator.Transpose(gradient, false);
           
            return (gradTransposed, null);
        }
    }
}
