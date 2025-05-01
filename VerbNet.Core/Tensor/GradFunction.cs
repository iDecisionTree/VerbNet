namespace VerbNet.Core
{
    public static class GradFunction
    {
        public static (Tensor, Tensor) AddGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            return (gradient, gradient);
        }

        public static (Tensor, Tensor) SubGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            Tensor rightGrad = TensorOperator.Negate(gradient, false);

            return (gradient, rightGrad);
        }

        public static (Tensor, Tensor) MulGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            Tensor leftGrad = TensorOperator.Multiply(gradient, rightLeaf, false);
            Tensor rightGrad = TensorOperator.Multiply(gradient, leftLeaf, false);

            return (leftGrad, rightGrad);
        }

        public static (Tensor, Tensor) DivGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            using Tensor rightSquared = TensorOperator.Multiply(rightLeaf, rightLeaf, false);
            using Tensor negLeft = TensorOperator.Negate(leftLeaf, false);
            using Tensor temp = TensorOperator.Multiply(negLeft, gradient, false);

            Tensor leftGrad = TensorOperator.Divide(gradient, rightLeaf, false);
            Tensor rightGrad = TensorOperator.Divide(temp, rightSquared, false);

            return (leftGrad, rightGrad);
        }

        public static (Tensor, Tensor) NegGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            Tensor leftGrad = TensorOperator.Negate(gradient, false);

            return (leftGrad, null);
        }

        public static (Tensor, Tensor) AbsGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            using Tensor sign = TensorOperator.Sign(leftLeaf);
            Tensor leftGrad = TensorOperator.Multiply(gradient, sign, false);

            return (leftGrad, null);
        }

        public static (Tensor, Tensor) SignGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            return (Tensor.Zero, null);
        }

        public static (Tensor, Tensor) SqrtGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            using Tensor sqrtX = TensorOperator.Sqrt(leftLeaf, false);
            using Tensor denominator = TensorOperator.Multiply(new Tensor([2f], [1], false), sqrtX, false);
            Tensor grad = TensorOperator.Divide(gradient, denominator, false);

            return (grad, null);
        }

        public static (Tensor, Tensor) LogEGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            using Tensor reciprocal = TensorOperator.Divide(Tensor.One, leftLeaf, false);
            Tensor grad = TensorOperator.Multiply(gradient, reciprocal, false);

            return (grad, null);
        }

        public static (Tensor, Tensor) ExpGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            using Tensor expX = TensorOperator.Exp(leftLeaf, false);
            Tensor grad = TensorOperator.Multiply(gradient, expX, false);

            return (grad, null);
        }

        public static (Tensor, Tensor) PowerGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            float exponent = (float)opArgs["Power_exponent"];
            using Tensor powerTensor = new Tensor([exponent], [1], false);

            using Tensor leftPower = TensorOperator.Power(leftLeaf, exponent - 1f, false);
            Tensor leftGrad = TensorOperator.Multiply(gradient, TensorOperator.Multiply(powerTensor, leftPower, false), false);

            return (leftGrad, null);
        }

        public static (Tensor, Tensor) SinGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            using Tensor cosX = TensorOperator.Cos(leftLeaf, false);
            Tensor grad = TensorOperator.Multiply(gradient, cosX, false);

            return (grad, null);
        }

        public static (Tensor, Tensor) CosGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            using Tensor sinX = TensorOperator.Sin(leftLeaf, false);
            using Tensor negSinX = TensorOperator.Negate(sinX, false);
            Tensor grad = TensorOperator.Multiply(gradient, negSinX, false);

            return (grad, null);
        }

        public static (Tensor, Tensor) TanGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            using Tensor cosX = TensorOperator.Cos(leftLeaf, false);
            using Tensor cosSquared = TensorOperator.Multiply(cosX, cosX, false);
            using Tensor invSecSquared = TensorOperator.Divide(Tensor.One, cosSquared, false);
            Tensor grad = TensorOperator.Multiply(gradient, invSecSquared, false);

            return (grad, null);
        }

        public static (Tensor, Tensor) SinhGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            using Tensor coshX = TensorOperator.Cosh(leftLeaf, false);
            Tensor grad = TensorOperator.Multiply(gradient, coshX, false);

            return (grad, null);
        }

        public static (Tensor, Tensor) CoshGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            using Tensor sinhX = TensorOperator.Sinh(leftLeaf, false);
            Tensor grad = TensorOperator.Multiply(gradient, sinhX, false);

            return (grad, null);
        }

        public static (Tensor, Tensor) TanhGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            using Tensor tanhX = TensorOperator.Tanh(leftLeaf, false);
            using Tensor tanhSquared = TensorOperator.Multiply(tanhX, tanhX, false);
            using Tensor oneMinusTanhSquared = TensorOperator.Subtract(Tensor.One, tanhSquared, false);
            Tensor grad = TensorOperator.Multiply(gradient, oneMinusTanhSquared, false);

            return (grad, null);
        }

        public static (Tensor, Tensor) MatMulGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            using Tensor bTransposed = TensorOperator.Transpose(rightLeaf, false);
            Tensor dA = TensorOperator.MatMul(gradient, bTransposed, false);

            using Tensor aTransposed = TensorOperator.Transpose(leftLeaf, false);
            Tensor dB = TensorOperator.MatMul(aTransposed, gradient, false);

            return (dA, dB);
        }

        public static (Tensor, Tensor) TransposeGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            Tensor grad = TensorOperator.Transpose(gradient, false);

            return (grad, null);
        }

        public static (Tensor, Tensor) RepeatGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            int axis = (int)opArgs["Repeat_axis"];
            int repeat = (int)opArgs["Repeat_repeat"];
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

        public static (Tensor, Tensor) ReshapeGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            int originalSize = leftLeaf.Shape.Aggregate(1, (a, b) => a * b);
            int reshapedSize = gradient.Shape.Aggregate(1, (a, b) => a * b);

            if (originalSize != reshapedSize)
            {
                throw new InvalidOperationException($"Shape mismatch: cannot reshape gradient {string.Join(",", gradient.Shape)} to original shape {string.Join(",", leftLeaf.Shape)}");
            }

            Tensor reshapedGradient = new Tensor(gradient.Data.Clone(), leftLeaf.Shape, gradient.RequiresGrad);

            return (reshapedGradient, null);
        }

        public static (Tensor, Tensor) BroadcastGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            int[] originalShape = (int[])opArgs["OriginalShape"];
            Tensor grad = TensorOperator.Reshape(gradient, originalShape, false);

            return (grad, null);
        }
    }
}