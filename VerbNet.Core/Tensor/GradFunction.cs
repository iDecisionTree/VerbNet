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
            byte[] maxIndices = (byte[])opArgs["Max_maxIndices"];

            AlignedArray<float> leftGradData = new AlignedArray<float>(leftLeaf.Length, leftLeaf.Data.Alignment);
            AlignedArray<float> rightGradData = new AlignedArray<float>(rightLeaf.Length, rightLeaf.Data.Alignment);

            for (int i = 0; i < maxIndices.Length; i++)
            {
                float grad = gradient.Data[i];
                switch (maxIndices[i])
                {
                    case 0:
                        leftGradData[i] = grad;
                        break;
                    case 1:
                        rightGradData[i] = grad;
                        break;
                    case 2:
                        leftGradData[i] = grad * 0.5f;
                        rightGradData[i] = grad * 0.5f;
                        break;
                }
            }

            Tensor leftGrad = new Tensor(leftGradData, leftLeaf.Shape, false);
            Tensor rightGrad = new Tensor(rightGradData, rightLeaf.Shape, false);

            return (leftGrad, rightGrad);
        }

        public static (Tensor, Tensor) MinGradFn(Tensor gradient, Tensor leftLeaf, Tensor rightLeaf, Dictionary<string, object> opArgs)
        {
            byte[] minIndices = (byte[])opArgs["Min_minIndices"];

            AlignedArray<float> leftGradData = new AlignedArray<float>(leftLeaf.Length, leftLeaf.Data.Alignment);
            AlignedArray<float> rightGradData = new AlignedArray<float>(rightLeaf.Length, rightLeaf.Data.Alignment);

            for (int i = 0; i < minIndices.Length; i++)
            {
                float grad = gradient.Data[i];
                switch (minIndices[i])
                {
                    case 0:
                        leftGradData[i] = grad;
                        break;
                    case 1:
                        rightGradData[i] = grad;
                        break;
                    case 2:
                        leftGradData[i] = grad * 0.5f;
                        rightGradData[i] = grad * 0.5f;
                        break;
                }
            }

            Tensor leftGrad = new Tensor(leftGradData, leftLeaf.Shape, false);
            Tensor rightGrad = new Tensor(rightGradData, rightLeaf.Shape, false);

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
    }
}