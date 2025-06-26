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
            Tensor rightGrad = TensorOperator.Negate(gradient, false, false);

            return (gradient, rightGrad);
        }

        public static (Tensor, Tensor) MulGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            Tensor leftGrad = TensorOperator.Multiply(gradient, rightLeaf, false, false);
            Tensor rightGrad = TensorOperator.Multiply(gradient, leftLeaf, false, false);

            return (leftGrad, rightGrad);
        }

        public static (Tensor, Tensor) DivGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            Tensor rightSquared = TensorOperator.Multiply(rightLeaf, rightLeaf, false, false);
            Tensor negLeft = TensorOperator.Negate(leftLeaf, false, false);
            Tensor temp = TensorOperator.Multiply(negLeft, gradient, false, false);

            Tensor leftGrad = TensorOperator.Divide(gradient, rightLeaf, false, false);
            Tensor rightGrad = TensorOperator.Divide(temp, rightSquared, false, false);

            return (leftGrad, rightGrad);
        }

        public static (Tensor, Tensor) NegGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            Tensor leftGrad = TensorOperator.Negate(gradient, false, false);

            return (leftGrad, null);
        }

        public static (Tensor, Tensor) AbsGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            Tensor sign = TensorOperator.Sign(leftLeaf, false, false);
            Tensor leftGrad = TensorOperator.Multiply(gradient, sign, false, false);

            return (leftGrad, null);
        }

        public static (Tensor, Tensor) SignGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            return (Tensor.Zero, null);
        }

        public static (Tensor, Tensor) SqrtGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            Tensor sqrtX = TensorOperator.Sqrt(leftLeaf, false, false);
            Tensor denominator = TensorOperator.Multiply(Tensor.Create(2f, false), sqrtX, false, false);
            Tensor grad = TensorOperator.Divide(gradient, denominator, false, false);

            return (grad, null);
        }

        public static (Tensor, Tensor) LogEGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            Tensor reciprocal = TensorOperator.Divide(Tensor.One, leftLeaf, false, false);
            Tensor grad = TensorOperator.Multiply(gradient, reciprocal, false, false);

            return (grad, null);
        }

        public static (Tensor, Tensor) ExpGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            Tensor expX = TensorOperator.Exp(leftLeaf, false, false);
            Tensor grad = TensorOperator.Multiply(gradient, expX, false, false);

            return (grad, null);
        }

        public static (Tensor, Tensor) PowerGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            float exponent = (float)opArgs["Power_exponent"];

            Tensor leftPower = TensorOperator.Power(leftLeaf, exponent - 1f, false, false);
            Tensor leftGrad = TensorOperator.Multiply(gradient, TensorOperator.Multiply(Tensor.Create(exponent, false), leftPower, false, false), false, false);

            return (leftGrad, null);
        }

        public static (Tensor, Tensor) SinGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            Tensor cosX = TensorOperator.Cos(leftLeaf, false, false);
            Tensor grad = TensorOperator.Multiply(gradient, cosX, false, false);

            return (grad, null);
        }

        public static (Tensor, Tensor) CosGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            Tensor sinX = TensorOperator.Sin(leftLeaf, false, false);
            Tensor negSinX = TensorOperator.Negate(sinX, false, false);
            Tensor grad = TensorOperator.Multiply(gradient, negSinX, false, false);

            return (grad, null);
        }

        public static (Tensor, Tensor) TanGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            Tensor cosX = TensorOperator.Cos(leftLeaf, false, false);
            Tensor cosSquared = TensorOperator.Multiply(cosX, cosX, false, false);
            Tensor invSecSquared = TensorOperator.Divide(Tensor.One, cosSquared, false, false);
            Tensor grad = TensorOperator.Multiply(gradient, invSecSquared, false, false);

            return (grad, null);
        }

        public static (Tensor, Tensor) SinhGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            Tensor coshX = TensorOperator.Cosh(leftLeaf, false, false);
            Tensor grad = TensorOperator.Multiply(gradient, coshX, false, false);

            return (grad, null);
        }

        public static (Tensor, Tensor) CoshGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            Tensor sinhX = TensorOperator.Sinh(leftLeaf, false, false);
            Tensor grad = TensorOperator.Multiply(gradient, sinhX, false, false);

            return (grad, null);
        }

        public static (Tensor, Tensor) TanhGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            Tensor tanhX = TensorOperator.Tanh(leftLeaf, false, false);
            Tensor tanhSquared = TensorOperator.Multiply(tanhX, tanhX, false, false);
            Tensor oneMinusTanhSquared = TensorOperator.Subtract(Tensor.One, tanhSquared, false, false);
            Tensor grad = TensorOperator.Multiply(gradient, oneMinusTanhSquared, false, false);

            return (grad, null);
        }

        public static (Tensor, Tensor) MaxGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            bool[] maskA = (bool[])opArgs["Max_maskA"]; 
            bool[] maskB = (bool[])opArgs["Max_maskB"];
            bool[] equalMask = (bool[])opArgs["Max_equalMask"];

            Tensor leftGrad = new Tensor(leftLeaf.Shape, false);
            Tensor rightGrad = new Tensor(rightLeaf.Shape, false);

            for (int i = 0; i < gradient.Length; i++)
            {
                int[] broadcastedIndex = TensorOperator.GetMultiIndex(i, gradient.Shape);

                if (maskA[i])
                {
                    int[] originalMultiIndex = TensorOperator.BroadcastedToOriginalIndex(broadcastedIndex, leftLeaf.Shape, gradient.Shape);
                    int originalIndex = TensorOperator.GetLinearIndex(originalMultiIndex, leftLeaf.Shape);
                    leftGrad.Data[originalIndex] += gradient.Data[i];
                }
                else if (maskB[i])
                {
                    int[] originalMultiIndex = TensorOperator.BroadcastedToOriginalIndex(broadcastedIndex, rightLeaf.Shape, gradient.Shape);
                    int originalIndex = TensorOperator.GetLinearIndex(originalMultiIndex, rightLeaf.Shape);
                    rightGrad.Data[originalIndex] += gradient.Data[i];
                }
                else if (equalMask[i])
                {
                    int[] originalMultiIndexLeft = TensorOperator.BroadcastedToOriginalIndex(broadcastedIndex, leftLeaf.Shape, gradient.Shape);
                    int originalIndexLeft = TensorOperator.GetLinearIndex(originalMultiIndexLeft, leftLeaf.Shape);
                    int[] originalMultiIndexRight = TensorOperator.BroadcastedToOriginalIndex(broadcastedIndex, rightLeaf.Shape, gradient.Shape);
                    int originalIndexRight = TensorOperator.GetLinearIndex(originalMultiIndexRight, rightLeaf.Shape);

                    leftGrad.Data[originalIndexLeft] += gradient.Data[i] * 0.5f;
                    rightGrad.Data[originalIndexRight] += gradient.Data[i] * 0.5f;
                }
            }

            return (leftGrad, rightGrad);
        }

        public static (Tensor, Tensor) MinGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            bool[] maskA = (bool[])opArgs["Min_maskA"];
            bool[] maskB = (bool[])opArgs["Min_maskB"];
            bool[] equalMask = (bool[])opArgs["Min_equalMask"];

            Tensor leftGrad = new Tensor(leftLeaf.Shape, false);
            Tensor rightGrad = new Tensor(rightLeaf.Shape, false);

            for (int i = 0; i < gradient.Length; i++)
            {
                int[] broadcastedIndex = TensorOperator.GetMultiIndex(i, gradient.Shape);

                if (maskA[i])
                {
                    int[] originalMultiIndex = TensorOperator.BroadcastedToOriginalIndex(broadcastedIndex, rightLeaf.Shape, gradient.Shape);
                    int originalIndex = TensorOperator.GetLinearIndex(originalMultiIndex, rightLeaf.Shape);
                    rightGrad.Data[originalIndex] += gradient.Data[i];
                }
                else if (maskB[i])
                {
                    int[] originalMultiIndex = TensorOperator.BroadcastedToOriginalIndex(broadcastedIndex, leftLeaf.Shape, gradient.Shape);
                    int originalIndex = TensorOperator.GetLinearIndex(originalMultiIndex, leftLeaf.Shape);
                    leftGrad.Data[originalIndex] += gradient.Data[i];
                }
                else if (equalMask[i])
                {
                    int[] originalMultiIndexLeft = TensorOperator.BroadcastedToOriginalIndex(broadcastedIndex, leftLeaf.Shape, gradient.Shape);
                    int originalIndexLeft = TensorOperator.GetLinearIndex(originalMultiIndexLeft, leftLeaf.Shape);
                    int[] originalMultiIndexRight = TensorOperator.BroadcastedToOriginalIndex(broadcastedIndex, rightLeaf.Shape, gradient.Shape);
                    int originalIndexRight = TensorOperator.GetLinearIndex(originalMultiIndexRight, rightLeaf.Shape);

                    leftGrad.Data[originalIndexLeft] += gradient.Data[i] * 0.5f;
                    rightGrad.Data[originalIndexRight] += gradient.Data[i] * 0.5f;
                }
            }

            return (leftGrad, rightGrad);
        }

        public static (Tensor, Tensor) FloorGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            return (Tensor.Zero, null);
        }

        public static (Tensor, Tensor) CeilingGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            return (Tensor.Zero, null);
        }

        public static (Tensor, Tensor) RoundGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            return (Tensor.Zero, null);
        }

        public static (Tensor, Tensor) MatMulGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            Tensor bTransposed = TensorOperator.Transpose(rightLeaf, false, false);
            Tensor dA = TensorOperator.MatMul(gradient, bTransposed, false, false);

            Tensor aTransposed = TensorOperator.Transpose(leftLeaf, false, false);
            Tensor dB = TensorOperator.MatMul(aTransposed, gradient, false, false);

            return (dA, dB);
        }

        public static (Tensor, Tensor) TransposeGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            Tensor grad = TensorOperator.Transpose(gradient, false, false);

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

        public static (Tensor, Tensor) ConcatGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            int dim = (int)opArgs["Concat_dim"];
            int aDim = leftLeaf.Shape[dim];
            int bDim = rightLeaf.Shape[dim];
            int totalElements = gradient.Length;

            AlignedArray<float> leftGradData = new AlignedArray<float>(leftLeaf.Length, gradient.Data.Alignment);
            AlignedArray<float> rightGradData = new AlignedArray<float>(rightLeaf.Length, gradient.Data.Alignment);

            Parallel.For(0, totalElements, i =>
            {
                int[] indices = TensorOperator.GetMultiIndex(i, gradient.Shape);
                int dimIndex = indices[dim];

                if (dimIndex < aDim)
                {
                    int[] leftIndices = (int[])indices.Clone();
                    int leftIndex = TensorOperator.GetLinearIndex(leftIndices, leftLeaf.Shape);
                    leftGradData[leftIndex] = gradient.Data[i];
                }
                else
                {
                    int[] rightIndices = (int[])indices.Clone();
                    rightIndices[dim] = dimIndex - aDim;
                    int rightIndex = TensorOperator.GetLinearIndex(rightIndices, rightLeaf.Shape);
                    rightGradData[rightIndex] = gradient.Data[i];
                }
            });

            Tensor leftGrad = new Tensor(leftGradData, leftLeaf.Shape, false);
            Tensor rightGrad = new Tensor(rightGradData, rightLeaf.Shape, false);

            return (leftGrad, rightGrad);
        }

        public static (Tensor, Tensor) BroadcastGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            int[] originalShape = (int[])opArgs["OriginalShape"];
            Tensor grad = TensorOperator.Reshape(gradient, originalShape, false, false);

            return (grad, null);
        }
    }
}