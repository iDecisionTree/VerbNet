namespace VerbNet.Core
{
    public static class TensorOperator
    {
        public static Tensor Add(Tensor a, Tensor b, bool buildGraph = true, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));
            if (b == null)
                throw new ArgumentNullException(nameof(b));

            Tensor aBroadcast, bBroadcast;
            (aBroadcast, bBroadcast) = Broadcast(a, b, false);

            bool requiresGrad = computeGrad && (a.RequiresGrad || b.RequiresGrad);
            Tensor result = new Tensor(Operator.Add(aBroadcast.Data, bBroadcast.Data), aBroadcast.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.AddGradFn;
                result.LeftLeaf = a;
                result.RightLeaf = b;
                result.LeftLeaf.Father = result;
                result.RightLeaf.Father = result;
            }
            else if (requiresGrad && computeGrad)
            {
                if (a.RequiresGrad && b.RequiresGrad)
                {
                    result.Gradient = Add(a.Gradient, b.Gradient, false, false);
                }
                else
                {
                    if (a.RequiresGrad)
                    {
                        result.Gradient = a.Gradient.Clone();
                    }
                    else if (b.RequiresGrad)
                    {
                        result.Gradient = b.Gradient.Clone();
                    }
                }
            }

            return result;
        }

        public static Tensor Add(Tensor a, float b, bool buildGraph = true)
        {
            Tensor bTensor = Tensor.Create(b);

            return Add(a, bTensor, false);
        }

        public static Tensor Subtract(Tensor a, Tensor b, bool buildGraph = true, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));
            if (b == null)
                throw new ArgumentNullException(nameof(b));

            Tensor aBroadcast, bBroadcast;
            (aBroadcast, bBroadcast) = Broadcast(a, b, false);

            bool requiresGrad = computeGrad && (a.RequiresGrad || b.RequiresGrad);
            Tensor result = new Tensor(Operator.Subtract(aBroadcast.Data, bBroadcast.Data), aBroadcast.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.SubGradFn;
                result.LeftLeaf = a;
                result.RightLeaf = b;
                result.LeftLeaf.Father = result;
                result.RightLeaf.Father = result;
            }
            else if (requiresGrad && computeGrad)
            {
                if (a.RequiresGrad && b.RequiresGrad)
                {
                    result.Gradient = Subtract(a.Gradient, b.Gradient, false, false);
                }
                else
                {
                    if (a.RequiresGrad)
                    {
                        result.Gradient = a.Gradient.Clone();
                    }
                    else if (b.RequiresGrad)
                    {
                        result.Gradient = b.Gradient.Clone();
                    }
                }
            }

            return result;
        }

        public static Tensor Subtract(Tensor a, float b)
        {
            Tensor bTensor = Tensor.Create(b);

            return Subtract(a, bTensor, false);
        }

        public static Tensor Subtract(float a, Tensor b)
        {
            Tensor aTensor = Tensor.Create(a);

            return Subtract(aTensor, b, false);
        }

        public static Tensor Multiply(Tensor a, Tensor b, bool buildGraph = true, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));
            if (b == null)
                throw new ArgumentNullException(nameof(b));

            Tensor aBroadcast, bBroadcast;
            (aBroadcast, bBroadcast) = Broadcast(a, b, false);

            bool requiresGrad = computeGrad && (a.RequiresGrad || b.RequiresGrad);
            Tensor result = new Tensor(Operator.Multiply(aBroadcast.Data, bBroadcast.Data), aBroadcast.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.MulGradFn;
                result.LeftLeaf = a;
                result.RightLeaf = b;
                result.LeftLeaf.Father = result;
                result.RightLeaf.Father = result;
            }
            else if (requiresGrad && computeGrad)
            {
                if (a.RequiresGrad && b.RequiresGrad)
                {
                    result.Gradient = Add(Multiply(a.Gradient, b, false, false), Multiply(a, b.Gradient, false, false), false, false);
                }
                else
                {
                    if (a.RequiresGrad)
                    {
                        result.Gradient = a.Gradient.Clone();
                    }
                    else if (b.RequiresGrad)
                    {
                        result.Gradient = b.Gradient.Clone();
                    }
                }
            }

            return result;
        }

        public static Tensor Multiply(Tensor a, float b)
        {
            Tensor bTensor = Tensor.Create(b);

            return Multiply(a, bTensor, false);
        }

        public static Tensor Divide(Tensor a, Tensor b, bool buildGraph = true, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));
            if (b == null)
                throw new ArgumentNullException(nameof(b));

            Tensor aBroadcast, bBroadcast;
            (aBroadcast, bBroadcast) = Broadcast(a, b, false);

            bool requiresGrad = computeGrad && (a.RequiresGrad || b.RequiresGrad);
            Tensor result = new Tensor(Operator.Divide(aBroadcast.Data, bBroadcast.Data), aBroadcast.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.DivGradFn;
                result.LeftLeaf = a;
                result.RightLeaf = b;
                result.LeftLeaf.Father = result;
                result.RightLeaf.Father = result;
            }
            else if (requiresGrad && computeGrad)
            {
                if (a.RequiresGrad && b.RequiresGrad)
                {
                    Tensor gradA = Divide(a.Gradient, b, false, false);
                    Tensor gradB = Negate(Divide(Multiply(a, b.Gradient, false, false), Power(b, 2, false), false, false), false, false);
                    result.Gradient = Add(gradA, gradB, false, false);
                }
                else
                {
                    if (a.RequiresGrad)
                    {
                        result.Gradient = a.Gradient.Clone();
                    }
                    else if (b.RequiresGrad)
                    {
                        result.Gradient = b.Gradient.Clone();
                    }
                }
            }

            return result;
        }

        public static Tensor Divide(Tensor a, float b)
        {
            Tensor bTensor = Tensor.Create(b);

            return Divide(a, bTensor, false);
        }

        public static Tensor Divide(float a, Tensor b)
        {
            Tensor aTensor = Tensor.Create(a);

            return Divide(aTensor, b, false);
        }

        public static Tensor Negate(Tensor a, bool buildGraph = true, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            bool requiresGrad = computeGrad && a.RequiresGrad;
            Tensor result = new Tensor(Operator.Negate(a.Data), a.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.NegGradFn;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }
            else if (requiresGrad && computeGrad)
            {
                result.Gradient = Negate(a.Gradient, false, false);
            }

            return result;
        }

        public static Tensor Abs(Tensor a, bool buildGraph = true, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            bool requiresGrad = computeGrad && a.RequiresGrad;
            Tensor result = new Tensor(Operator.Abs(a.Data), a.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.AbsGradFn;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }
            else if (requiresGrad && computeGrad)
            {
                Tensor sign = Sign(a.Gradient, false, false);
                result.Gradient = Multiply(Tensor.One, sign, false, false);
            }

            return result;
        }

        public static Tensor Sign(Tensor a, bool buildGraph = true, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            bool requiresGrad = computeGrad && a.RequiresGrad;
            Tensor result = new Tensor(Operator.Sign(a.Data), a.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.SignGradFn;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }
            else if (requiresGrad && computeGrad)
            {
                result.Gradient = Tensor.Zero;
            }

            return result;
        }

        public static Tensor Sqrt(Tensor a, bool buildGraph = true, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            bool requiresGrad = computeGrad && a.RequiresGrad;
            Tensor result = new Tensor(Operator.Sqrt(a.Data), a.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.SqrtGradFn;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }
            else if (requiresGrad && computeGrad)
            {
                Tensor sqrtX = Sqrt(a.Gradient, false, false);
                Tensor denominator = Multiply(Tensor.Create(2f, false), sqrtX, false, false);
                result.Gradient = Divide(Tensor.One, denominator, false, false);
            }

            return result;
        }

        public static Tensor LogE(Tensor a, bool buildGraph = true, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            bool requiresGrad = computeGrad && a.RequiresGrad;
            Tensor result = new Tensor(Operator.LogE(a.Data), a.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.LogEGradFn;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }
            else if (requiresGrad && computeGrad)
            {
                result.Gradient = Divide(Tensor.One, a.Gradient, false, false);
            }

            return result;
        }

        public static Tensor Exp(Tensor a, bool buildGraph = true, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            bool requiresGrad = computeGrad && a.RequiresGrad;
            Tensor result = new Tensor(Operator.Exp(a.Data), a.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.ExpGradFn;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }
            else if (requiresGrad && computeGrad)
            {
                result.Gradient = Exp(a.Gradient, false, false);
            }

            return result;
        }

        public static Tensor Power(Tensor a, float exponent, bool buildGraph = true, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            bool requiresGrad = computeGrad && a.RequiresGrad;
            Tensor result = new Tensor(Operator.Power(a.Data, exponent), a.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.PowerGradFn;
                result.OpArgs.Add("Power_exponent", exponent);
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }
            else if (requiresGrad && computeGrad)
            {
                Tensor power = Power(a.Gradient, exponent - 1f);
                result.Gradient = Multiply(power, Tensor.Create(exponent, false), false, false);
            }

            return result;
        }

        public static Tensor Sin(Tensor a, bool buildGraph = true, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            bool requiresGrad = computeGrad && a.RequiresGrad;
            Tensor result = new Tensor(Operator.Sin(a.Data), a.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.SinGradFn;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }
            else if (requiresGrad && computeGrad)
            {
                result.Gradient = Cos(a.Gradient, false, false);
            }

            return result;
        }

        public static Tensor Cos(Tensor a, bool buildGraph = true, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            bool requiresGrad = computeGrad && a.RequiresGrad;
            Tensor result = new Tensor(Operator.Cos(a.Data), a.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.CosGradFn;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }
            else if (requiresGrad && computeGrad)
            {
                Tensor sinX = Sin(a.Gradient, false, false);
                result.Gradient = Negate(sinX, false, false);
            }

            return result;
        }

        public static Tensor Tan(Tensor a, bool buildGraph = true, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            bool requiresGrad = computeGrad && a.RequiresGrad;
            Tensor result = new Tensor(Operator.Tan(a.Data), a.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.TanGradFn;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }
            else if (requiresGrad && computeGrad)
            {
                Tensor cosX = Cos(a.Gradient, false, false);
                Tensor cosSquared = Multiply(cosX, cosX, false, false);
                result.Gradient = Divide(Tensor.One, cosSquared, false, false);
            }

            return result;
        }

        public static Tensor Sinh(Tensor a, bool buildGraph = true, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            bool requiresGrad = computeGrad && a.RequiresGrad;
            Tensor result = new Tensor(Operator.Sinh(a.Data), a.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.SinhGradFn;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }
            else if (requiresGrad && computeGrad)
            {
                result.Gradient = Cosh(a.Gradient, false, false);
            }

            return result;
        }

        public static Tensor Cosh(Tensor a, bool buildGraph = true, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            bool requiresGrad = computeGrad && a.RequiresGrad;
            Tensor result = new Tensor(Operator.Cosh(a.Data), a.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.CoshGradFn;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }
            else if (requiresGrad && computeGrad)
            {
                result.Gradient = Sinh(a.Gradient, false, false);
            }

            return result;
        }

        public static Tensor Tanh(Tensor a, bool buildGraph = true, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            bool requiresGrad = computeGrad && a.RequiresGrad;
            Tensor result = new Tensor(Operator.Tanh(a.Data), a.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.TanhGradFn;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }
            else if (requiresGrad && computeGrad)
            {
                Tensor tanhX = Tanh(a.Gradient, false, false);
                Tensor tanhSquared = Multiply(tanhX, tanhX, false, false);
                result.Gradient = Subtract(Tensor.One, tanhSquared, false, false);
            }

            return result;
        }

        public static Tensor Max(Tensor a, Tensor b, bool buildGraph = true, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));
            if (b == null)
                throw new ArgumentNullException(nameof(b));

            Tensor aBroadcast, bBroadcast;
            (aBroadcast, bBroadcast) = Broadcast(a, b, false);

            bool requiresGrad = computeGrad && (a.RequiresGrad || b.RequiresGrad);
            Tensor result = new Tensor(Operator.Max(aBroadcast.Data, bBroadcast.Data), aBroadcast.Shape, requiresGrad);

            if (buildGraph)
            {
                byte[] maxIndices = new byte[aBroadcast.Length];
                for (int i = 0; i < aBroadcast.Length; i++)
                {
                    float aValue = aBroadcast.Data[i];
                    float bValue = bBroadcast.Data[i];

                    if (aValue > bValue)
                    {
                        maxIndices[i] = 0;
                    }
                    else if (aValue < bValue)
                    {
                        maxIndices[i] = 1;
                    }
                    else
                    {
                        maxIndices[i] = 2;
                    }
                }

                result.GradFn = GradFunction.MaxGradFn;
                result.OpArgs.Add("Max_maxIndices", maxIndices);
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
                result.RightLeaf = b;
                result.RightLeaf.Father = result;
            }
            else if (requiresGrad && computeGrad)
            {
                result.Gradient = Max(a.Gradient, b.Gradient, false, false);
            }

            return result;
        }

        public static Tensor Max(Tensor a, float b)
        {
            Tensor bTensor = Tensor.Create(b);

            return Max(a, bTensor, false);
        }

        public static Tensor Max(float a, Tensor b)
        {
            Tensor aTensor = Tensor.Create(a);

            return Max(aTensor, b, false);
        }

        public static Tensor Min(Tensor a, Tensor b, bool buildGraph = true, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));
            if (b == null)
                throw new ArgumentNullException(nameof(b));

            Tensor aBroadcast, bBroadcast;
            (aBroadcast, bBroadcast) = Broadcast(a, b, false);

            bool requiresGrad = computeGrad && (a.RequiresGrad || b.RequiresGrad);
            Tensor result = new Tensor(Operator.Min(aBroadcast.Data, bBroadcast.Data), aBroadcast.Shape, requiresGrad);

            if (buildGraph)
            {
                byte[] minIndices = new byte[aBroadcast.Length];
                for (int i = 0; i < aBroadcast.Length; i++)
                {
                    float aValue = aBroadcast.Data[i];
                    float bValue = bBroadcast.Data[i];

                    if (aValue < bValue)
                    {
                        minIndices[i] = 0;
                    }
                    else if (aValue > bValue)
                    {
                        minIndices[i] = 1;
                    }
                    else
                    {
                        minIndices[i] = 2;
                    }
                }

                result.GradFn = GradFunction.MinGradFn;
                result.OpArgs.Add("Min_minIndices", minIndices);
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
                result.RightLeaf = b;
                result.RightLeaf.Father = result;
            }
            else if (requiresGrad && computeGrad)
            {
                result.Gradient = Min(a.Gradient, b.Gradient, false, false);
            }

            return result;
        }

        public static Tensor Min(Tensor a, float b)
        {
            Tensor bTensor = Tensor.Create(b);

            return Min(a, bTensor, false);
        }

        public static Tensor Min(float a, Tensor b)
        {
            Tensor aTensor = Tensor.Create(a);

            return Min(aTensor, b, false);
        }

        public static Tensor Floor(Tensor a, bool buildGraph = true, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            bool requiresGrad = computeGrad && a.RequiresGrad;
            Tensor result = new Tensor(Operator.Floor(a.Data), a.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.FloorGradFn;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }
            else if (requiresGrad && computeGrad)
            {
                result.Gradient = Tensor.Zero;
            }

            return result;
        }

        public static Tensor Ceiling(Tensor a, bool buildGraph = true, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            bool requiresGrad = computeGrad && a.RequiresGrad;
            Tensor result = new Tensor(Operator.Ceiling(a.Data), a.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.CeilingGradFn;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }
            else if (requiresGrad && computeGrad)
            {
                result.Gradient = Tensor.Zero;
            }

            return result;
        }

        public static Tensor Round(Tensor a, bool buildGraph = true, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            bool requiresGrad = computeGrad && a.RequiresGrad;
            Tensor result = new Tensor(Operator.Round(a.Data), a.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.RoundGradFn;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }
            else if (requiresGrad && computeGrad)
            {
                result.Gradient = Tensor.Zero;
            }

            return result;
        }

        public static Tensor MatMul(Tensor a, Tensor b, bool buildGraph = true, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));
            if (b == null)
                throw new ArgumentNullException(nameof(b));
            if (a.Rank != 2)
                throw new ArgumentException("Tensor a must be 2D for matrix multiplication.");
            if (b.Rank != 2)
                throw new ArgumentException("Tensor b must be 2D for matrix multiplication.");
            if (a.Shape[1] != b.Shape[0])
                throw new ArgumentException($"Cannot multiply matrices with shapes [{a.Shape[0]}, {a.Shape[1]}] and [{b.Shape[0]}, {b.Shape[1]}].");

            int aRows = a.Shape[0];
            int aCols = a.Shape[1];
            int bRows = b.Shape[0];
            int bCols = b.Shape[1];

            bool requiresGrad = computeGrad && (a.RequiresGrad || b.RequiresGrad);
            Tensor result = new Tensor(Operator.MatMul(a.Data, b.Data, aRows, aCols, bRows, bCols), [aRows, bCols], requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.MatMulGradFn;
                result.LeftLeaf = a;
                result.RightLeaf = b;
                result.LeftLeaf.Father = result;
                result.RightLeaf.Father = result;
            }
            else if (requiresGrad && computeGrad)
            {
                if (a.RequiresGrad && b.RequiresGrad)
                {
                    result.Gradient = MatMul(a.Gradient, b.Gradient, false, false);
                }
                else
                {
                    if (a.RequiresGrad)
                    {
                        result.Gradient = a.Gradient.Clone();
                    }
                    else if (b.RequiresGrad)
                    {
                        result.Gradient = b.Gradient.Clone();
                    }
                }
            }

            return result;
        }

        public static Tensor Transpose(Tensor a, bool buildGraph = true, bool computeGrad = true)
        {
            if (a.Rank != 2)
                throw new InvalidOperationException("Transpose requires a 2D tensor.");

            int rows = a.Shape[0];
            int cols = a.Shape[1];

            bool requiresGrad = computeGrad && a.RequiresGrad;
            Tensor result = new Tensor(Operator.Transpose(a.Data, rows, cols), [cols, rows], requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.TransposeGradFn;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }
            else if (requiresGrad && computeGrad)
            {
                result.Gradient = Transpose(a.Gradient, false, false);
            }

            return result;
        }

        public static Tensor Repeat(Tensor a, int axis, int repeat = 2, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));
            if (axis < 0 || axis >= a.Rank)
                throw new ArgumentOutOfRangeException(nameof(axis), "Axis must be within the tensor's rank.");
            if (repeat < 1)
                throw new ArgumentOutOfRangeException(nameof(repeat), "Repeat must be at least 1.");

            Tensor result;
            int[] newShape = a.Shape.ToArray();
            newShape[axis] *= repeat;

            if (repeat == 1)
            {
                result = Reshape(a, newShape, false);

                return result;
            }

            int outer = 1;
            for (int i = 0; i < axis; i++)
            {
                outer *= a.Shape[i];
            }

            int dim = a.Shape[axis];
            int inner = 1;
            for (int i = axis + 1; i < a.Rank; i++)
            {
                inner *= a.Shape[i];
            }

            AlignedArray<float> newData = new AlignedArray<float>(outer * dim * repeat * inner, a.Data.Alignment);

            Parallel.For(0, outer, o =>
            {
                int sourceOuterOffset = o * dim * inner;
                int targetOuterOffset = o * dim * repeat * inner;

                for (int d = 0; d < dim; d++)
                {
                    int sourceStart = sourceOuterOffset + d * inner;
                    int targetStartBase = targetOuterOffset + d * repeat * inner;

                    for (int i = 0; i < inner; i++)
                    {
                        newData[targetStartBase + i] = a.Data[sourceStart + i];
                    }

                    int copied = 1;
                    while (copied < repeat)
                    {
                        int toCopy = Math.Min(copied, repeat - copied);
                        for (int i = 0; i < toCopy * inner; i++)
                        {
                            newData[targetStartBase + copied * inner + i] = newData[targetStartBase + i];
                        }
                        copied += toCopy;
                    }
                }
            });

            bool requiresGrad = computeGrad && a.RequiresGrad;
            result = new Tensor(newData, newShape, requiresGrad);
            if (requiresGrad)
            {
                result.Gradient = Repeat(a.Gradient, axis, repeat, false);
            }

            return result;
        }

        public static Tensor Reshape(Tensor a, int[] newShape, bool computeGrad = true)
        {
            int length = a.Data.Length;
            int newLength = newShape.Aggregate(1, (a, b) => a * b);

            if (length != newLength)
                throw new ArgumentException($"Cannot reshape tensor from [{string.Join(", ", a.Shape)}] to [{string.Join(", ", newShape)}]");

            bool requiresGrad = computeGrad && a.RequiresGrad;
            Tensor result = new Tensor(a.Data, newShape, requiresGrad);
            if (requiresGrad)
            {
                result.Gradient = Reshape(a.Gradient, newShape, false);
            }

            return result;
        }

        public static Tensor Random(int[] shape, float scale = 1f, bool requiresGrad = false, string name = "")
        {
            if (shape == null)
                throw new ArgumentNullException(nameof(shape));
            if (shape.Length == 0)
                throw new ArgumentException("Shape must not be empty.", nameof(shape));
            foreach (int dim in shape)
            {
                if (dim <= 0)
                    throw new ArgumentException("All dimensions must be greater than 0.", nameof(shape));
            }

            int totalElements = shape.Aggregate(1, (a, b) => a * b);
            float[] data = new float[totalElements];
            Random rand = new Random();
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = (rand.NextSingle() * 2f - 1f) * scale;
            }

            Tensor result = new Tensor(data, shape, requiresGrad, name);

            return result;
        }

        public static int[] GetBroadcastShape(int[] aShape, int[] bShape)
        {
            List<int> reversedResult = new List<int>();
            int aLen = aShape.Length;
            int bLen = bShape.Length;
            int maxLen = Math.Max(aLen, bLen);

            for (int i = 0; i < maxLen; i++)
            {
                int aDim = (i < aLen) ? aShape[aLen - 1 - i] : 1;
                int bDim = (i < bLen) ? bShape[bLen - 1 - i] : 1;

                if (aDim == 1)
                {
                    reversedResult.Add(bDim);
                }
                else if (bDim == 1)
                {
                    reversedResult.Add(aDim);
                }
                else if (aDim == bDim)
                {
                    reversedResult.Add(aDim);
                }
                else
                {
                    throw new InvalidOperationException($"Shapes {string.Join(",", aShape)} and {string.Join(",", bShape)} cannot be broadcasted");
                }
            }

            reversedResult.Reverse();
            return reversedResult.ToArray();
        }

        public static Tensor BroadcastTo(Tensor tensor, int[] targetShape, bool computeGrad = true)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));
            if (targetShape == null)
                throw new ArgumentNullException(nameof(targetShape));

            int[] broadcastedShape = GetBroadcastShape(tensor.Shape, targetShape);
            if (!Enumerable.SequenceEqual(broadcastedShape, targetShape))
                throw new ArgumentException($"Cannot broadcast tensor from shape [{string.Join(", ", tensor.Shape)}] to [{string.Join(", ", targetShape)}]");

            int targetRank = targetShape.Length;
            int tensorRank = tensor.Shape.Length;
            int[] paddedTensorShape = new int[targetRank];
            for (int i = 0; i < targetRank; i++)
            {
                paddedTensorShape[i] = (i < targetRank - tensorRank) ? 1 : tensor.Shape[i - (targetRank - tensorRank)];
            }

            int[] tensorStrides = new int[targetRank];
            int stride = 1;
            for (int i = targetRank - 1; i >= 0; i--)
            {
                tensorStrides[i] = paddedTensorShape[i] == 1 ? 0 : stride;
                stride *= paddedTensorShape[i];
            }

            int[] targetStrides = new int[targetRank];
            stride = 1;
            for (int i = targetRank - 1; i >= 0; i--)
            {
                targetStrides[i] = stride;
                stride *= targetShape[i];
            }

            int totalElements = targetShape.Aggregate(1, (a, b) => a * b);
            AlignedArray<float> broadcastData = new AlignedArray<float>(totalElements, tensor.Data.Alignment);

            Parallel.For(0, totalElements, linearIndex =>
            {
                int originalIndex = 0;
                int temp = linearIndex;

                for (int dim = targetRank - 1; dim >= 0; dim--)
                {
                    int coord = temp % targetShape[dim];
                    temp /= targetShape[dim];

                    if (paddedTensorShape[dim] != 1)
                    {
                        originalIndex += (coord % paddedTensorShape[dim]) * tensorStrides[dim];
                    }
                }

                broadcastData[linearIndex] = tensor.Data[originalIndex];
            });

            bool requiresGrad = computeGrad && tensor.RequiresGrad;
            Tensor result = new Tensor(broadcastData, targetShape, requiresGrad);
            if (requiresGrad)
            {
                result.Gradient = BroadcastTo(tensor.Gradient, targetShape);
            }

            return result;
        }

        public static (Tensor, Tensor) Broadcast(Tensor a, Tensor b, bool computeGrad = true)
        {
            int[] broadcastShape = GetBroadcastShape(a.Shape, b.Shape);
            Tensor aBroadcast = BroadcastTo(a, broadcastShape, computeGrad);
            Tensor bBroadcast = BroadcastTo(b, broadcastShape, computeGrad);

            return (aBroadcast, bBroadcast);
        }
    }
}