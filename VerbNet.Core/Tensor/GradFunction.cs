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

        public static (Tensor, Tensor) AbsGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf)
        {
            return (TensorOperator.Multiply(gradient, TensorOperator.Sign(leftLeaf), false), null);
        }

        public static (Tensor, Tensor) SignGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf)
        {
            return (Tensor.Zero, null);
        }

        public static (Tensor, Tensor) SinGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf)
        {
            Tensor cosX = TensorOperator.Cos(leftLeaf, false);
            Tensor grad = TensorOperator.Multiply(gradient, cosX, false);

            return (grad, null);
        }

        public static (Tensor, Tensor) CosGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf)
        {
            Tensor sinX = TensorOperator.Sin(leftLeaf, false);
            Tensor negSinX = TensorOperator.Negate(sinX, false);
            Tensor grad = TensorOperator.Multiply(gradient, negSinX, false);

            return (grad, null);
        }

        public static (Tensor, Tensor) TanGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf)
        {
            Tensor cosX = TensorOperator.Cos(leftLeaf, false);
            Tensor secSquared = TensorOperator.Multiply(cosX, cosX, false);
            Tensor invSecSquared = TensorOperator.Divide(Tensor.One, secSquared, false);  // 1/cos²(x)
            Tensor grad = TensorOperator.Multiply(gradient, invSecSquared, false);

            return (grad, null);
        }

        public static (Tensor, Tensor) SinhGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf)
        {
            Tensor coshX = TensorOperator.Cosh(leftLeaf, false);
            Tensor grad = TensorOperator.Multiply(gradient, coshX, false);

            return (grad, null);
        }

        public static (Tensor, Tensor) CoshGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf)
        {
            Tensor sinhX = TensorOperator.Sinh(leftLeaf, false);
            Tensor grad = TensorOperator.Multiply(gradient, sinhX, false);

            return (grad, null);
        }

        public static (Tensor, Tensor) TanhGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf)
        {
            Tensor tanhX = TensorOperator.Tanh(leftLeaf, false);
            Tensor tanhSquared = TensorOperator.Multiply(tanhX, tanhX, false);
            Tensor oneMinusTanhSquared = TensorOperator.Subtract(Tensor.One, tanhSquared, false);
            Tensor grad = TensorOperator.Multiply(gradient, oneMinusTanhSquared, false);
            
            return (grad, null);
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
            return (TensorOperator.Transpose(gradient, false), null);
        }

        public static (Tensor, Tensor) RepeatGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf)
        {
            int axis = (int)gradient.OpArgs["Repeat_axis"];
            int repeat = (int)gradient.OpArgs["Repeat_repeat"];

            int[] originalShape = leftLeaf.Shape;

            int outer = 1;
            for (int i = 0; i < axis; i++)
            {
                outer *= originalShape[i];
            }

            int dim = originalShape[axis];
            int inner = 1;
            for (int i = axis + 1; i < originalShape.Length; i++)
            {
                inner *= originalShape[i];
            }

            float[] gradData = new float[leftLeaf.Data.Length];

            Parallel.For(0, outer, o =>
            {
                int sourceOuterOffset = o * dim * inner * repeat;
                int targetOuterOffset = o * dim * inner;

                for (int d = 0; d < dim; d++)
                {
                    int sourceStart = sourceOuterOffset + d * inner * repeat;
                    int targetStart = targetOuterOffset + d * inner;

                    for (int r = 0; r < repeat; r++)
                    {
                        int sourceOffset = sourceStart + r * inner;
                        for (int i = 0; i < inner; i++)
                        {
                            gradData[targetStart + i] += gradient.Data[sourceOffset + i];
                        }
                    }
                }
            });

            Tensor grad = new Tensor(gradData, originalShape, false);

            return (grad, null);
        }

        public static (Tensor, Tensor) ReshapeGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf)
        {
            if (gradient == null || leftLeaf == null)
                throw new ArgumentNullException("Gradient and input tensor cannot be null.");

            int originalSize = leftLeaf.Shape.Aggregate(1, (a, b) => a * b);
            int reshapedSize = gradient.Shape.Aggregate(1, (a, b) => a * b);

            if (originalSize != reshapedSize)
                throw new InvalidOperationException($"Shape mismatch: cannot reshape gradient {string.Join(",", gradient.Shape)} to original shape {string.Join(",", leftLeaf.Shape)}");

            Tensor reshapedGradient = new Tensor((float[])gradient.Data.Clone(), leftLeaf.Shape, gradient.RequiresGrad);

            return (reshapedGradient, null);
        }

        public static (Tensor, Tensor) BroadcastGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf)
        {
            int[] originalShape = (int[])gradient.OpArgs["OriginalShape"];

            return (TensorOperator.Reshape(gradient, originalShape, false), null);
        }
    }
}
