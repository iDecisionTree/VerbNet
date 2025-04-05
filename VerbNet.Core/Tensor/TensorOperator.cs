namespace VerbNet.Core
{
    public static class TensorOperator
    {
        public static Tensor Add(Tensor a, Tensor b, bool buildGraph = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));
            if (b == null)
                throw new ArgumentNullException(nameof(b));

            Tensor aBroadcast, bBroadcast;
            (aBroadcast, bBroadcast) = Broadcast(a, b, false);

            bool requiresGrad = buildGraph && (a.RequiresGrad || b.RequiresGrad);
            Tensor result = new Tensor(Operator.Add(aBroadcast.Data, bBroadcast.Data), aBroadcast.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.AddGradFn;
                result.LeftLeaf = a;
                result.RightLeaf = b;
                result.LeftLeaf.Father = result;
                result.RightLeaf.Father = result;
            }

            return result;
        }

        public static Tensor Subtract(Tensor a, Tensor b, bool buildGraph = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));
            if (b == null)
                throw new ArgumentNullException(nameof(b));

            Tensor aBroadcast, bBroadcast;
            (aBroadcast, bBroadcast) = Broadcast(a, b, false);

            bool requiresGrad = buildGraph && (a.RequiresGrad || b.RequiresGrad);
            Tensor result = new Tensor(Operator.Subtract(aBroadcast.Data, bBroadcast.Data), aBroadcast.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.SubGradFn;
                result.LeftLeaf = a;
                result.RightLeaf = b;
                result.LeftLeaf.Father = result;
                result.RightLeaf.Father = result;
            }

            return result;
        }

        public static Tensor Multiply(Tensor a, Tensor b, bool buildGraph = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));
            if (b == null)
                throw new ArgumentNullException(nameof(b));

            Tensor aBroadcast, bBroadcast;
            (aBroadcast, bBroadcast) = Broadcast(a, b, false);

            bool requiresGrad = buildGraph && (a.RequiresGrad || b.RequiresGrad);
            Tensor result = new Tensor(Operator.Multiply(aBroadcast.Data, bBroadcast.Data), aBroadcast.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.MulGradFn;
                result.LeftLeaf = a;
                result.RightLeaf = b;
                result.LeftLeaf.Father = result;
                result.RightLeaf.Father = result;
            }

            return result;
        }

        public static Tensor Divide(Tensor a, Tensor b, bool buildGraph = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));
            if (b == null)
                throw new ArgumentNullException(nameof(b));

            Tensor aBroadcast, bBroadcast;
            (aBroadcast, bBroadcast) = Broadcast(a, b, false);

            bool requiresGrad = buildGraph && (a.RequiresGrad || b.RequiresGrad);
            Tensor result = new Tensor(Operator.Divide(aBroadcast.Data, bBroadcast.Data), aBroadcast.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.DivGradFn;
                result.LeftLeaf = a;
                result.RightLeaf = b;
                result.LeftLeaf.Father = result;
                result.RightLeaf.Father = result;
            }

            return result;
        }

        public static Tensor Negate(Tensor a, bool buildGraph = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            bool requiresGrad = buildGraph && a.RequiresGrad;
            Tensor result = new Tensor(Operator.Negate(a.Data), a.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.NegGradFn;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }

            return result;
        }

        public static Tensor Abs(Tensor a, bool buildGraph = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            bool requiresGrad = buildGraph && a.RequiresGrad;
            Tensor result = new Tensor(Operator.Abs(a.Data), a.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.AbsGradFn;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }

            return result;
        }

        public static Tensor Sign(Tensor a, bool buildGraph = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            bool requiresGrad = buildGraph && a.RequiresGrad;
            Tensor result = new Tensor(Operator.Sign(a.Data), a.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.SignGradFn;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }

            return result;
        }

        public static Tensor Sqrt(Tensor a, bool buildGraph = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            bool requiresGrad = buildGraph && a.RequiresGrad;
            Tensor result = new Tensor(Operator.Sqrt(a.Data), a.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.SqrtGradFn;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }

            return result;
        }

        public static Tensor LogE(Tensor a, bool buildGraph = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            bool requiresGrad = buildGraph && a.RequiresGrad;
            Tensor result = new Tensor(Operator.LogE(a.Data), a.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.LogEGradFn;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }

            return result;
        }

        public static Tensor Sin(Tensor a, bool buildGraph = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            bool requiresGrad = buildGraph && a.RequiresGrad;
            Tensor result = new Tensor(Operator.Sin(a.Data), a.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.SinGradFn;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }

            return result;
        }

        public static Tensor Cos(Tensor a, bool buildGraph = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            bool requiresGrad = buildGraph && a.RequiresGrad;
            Tensor result = new Tensor(Operator.Cos(a.Data), a.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.CosGradFn;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }

            return result;
        }

        public static Tensor Tan(Tensor a, bool buildGraph = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            bool requiresGrad = buildGraph && a.RequiresGrad;
            Tensor result = new Tensor(Operator.Tan(a.Data), a.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.TanGradFn;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }

            return result;
        }

        public static Tensor Sinh(Tensor a, bool buildGraph = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            bool requiresGrad = buildGraph && a.RequiresGrad;
            Tensor result = new Tensor(Operator.Sinh(a.Data), a.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.SinhGradFn;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }

            return result;
        }

        public static Tensor Cosh(Tensor a, bool buildGraph = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            bool requiresGrad = buildGraph && a.RequiresGrad;
            Tensor result = new Tensor(Operator.Cosh(a.Data), a.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.CoshGradFn;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }

            return result;
        }

        public static Tensor Tanh(Tensor a, bool buildGraph = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            bool requiresGrad = buildGraph && a.RequiresGrad;
            Tensor result = new Tensor(Operator.Tanh(a.Data), a.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.TanhGradFn;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }

            return result;
        }

        public static Tensor MatMul(Tensor a, Tensor b, bool buildGraph = true)
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
            int bCols = b.Shape[1];

            bool requiresGrad = buildGraph && (a.RequiresGrad || b.RequiresGrad);
            Tensor result = new Tensor(Operator.MatMul(a.Data, b.Data, aRows, aCols, bCols), [aRows, bCols], requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.MatMulGradFn;
                result.LeftLeaf = a;
                result.RightLeaf = b;
                result.LeftLeaf.Father = result;
                result.RightLeaf.Father = result;
            }

            return result;
        }

        public static Tensor Transpose(Tensor a, bool buildGraph = true)
        {
            if (a.Rank != 2)
                throw new InvalidOperationException("Transpose requires a 2D tensor.");

            int rows = a.Shape[0];
            int cols = a.Shape[1];

            bool requiresGrad = buildGraph && a.RequiresGrad;
            Tensor result = new Tensor(Operator.Transpose(a.Data, rows, cols), [cols, rows], requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.TransposeGradFn;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }

            return result;
        }

        public static Tensor Repeat(Tensor a, int axis, int repeat = 2, bool buildGraph = true)
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

                if (buildGraph)
                {
                    result.GradFn = GradFunction.RepeatGradFn;
                    result.OpArgs.Add("Repeat_axis", axis);
                    result.OpArgs.Add("Repeat_repeat", repeat);
                    result.LeftLeaf = a;
                    result.LeftLeaf.Father = result;
                }

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

            float[] newData = new float[outer * dim * repeat * inner];
            Parallel.For(0, outer, o =>
            {
                int sourceOuterOffset = o * dim * inner;
                int targetOuterOffset = o * dim * repeat * inner;

                for (int d = 0; d < dim; d++)
                {
                    int sourceStart = sourceOuterOffset + d * inner;
                    int targetStartBase = targetOuterOffset + d * repeat * inner;

                    Array.Copy(a.Data, sourceStart, newData, targetStartBase, inner);

                    int copied = 1;
                    while (copied < repeat)
                    {
                        int toCopy = Math.Min(copied, repeat - copied);
                        Array.Copy(newData, targetStartBase, newData, targetStartBase + copied * inner, toCopy * inner);
                        copied += toCopy;
                    }
                }
            });

            bool requiresGrad = a.RequiresGrad;
            result = new Tensor(newData, newShape, requiresGrad);

            if (buildGraph && requiresGrad)
            {
                result.GradFn = GradFunction.RepeatGradFn;
                result.OpArgs.Add("Repeat_axis", axis);
                result.OpArgs.Add("Repeat_repeat", repeat);
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }

            return result;
        }

        public static Tensor Reshape(Tensor a, int[] newShape, bool buildGraph = true)
        {
            int length = a.Data.Length;
            int newLength = newShape.Aggregate(1, (a, b) => a * b);

            if (length != newLength)
                throw new ArgumentException($"Cannot reshape tensor from [{string.Join(", ", a.Shape)}] to [{string.Join(", ", newShape)}]");

            bool requiresGrad = a.RequiresGrad;
            Tensor result = new Tensor(a.Data, newShape, requiresGrad);

            if (buildGraph && requiresGrad)
            {
                result.GradFn = GradFunction.ReshapeGradFn;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }

            return result;
        }

        public static Tensor Random(int[] shape, bool requiresGrad = false, float scale = 1f)
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
                data[i] = (float)(rand.NextDouble() * 2d - 1d) * scale;
            }

            Tensor result = new Tensor(data, shape, requiresGrad);

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

        public static Tensor BroadcastTo(Tensor tensor, int[] targetShape, bool buildGraph = true)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));
            if (targetShape == null)
                throw new ArgumentNullException(nameof(targetShape));

            int[] broadcastedShape = GetBroadcastShape(tensor.Shape, targetShape);
            if (!Enumerable.SequenceEqual(broadcastedShape, targetShape))
                throw new ArgumentException($"Cannot broadcast tensor from shape [{string.Join(", ", tensor.Shape)}] to [{string.Join(", ", targetShape)}]");

            if (tensor.Shape.SequenceEqual(targetShape))
                return buildGraph ? tensor.Copy() : tensor;

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
            float[] broadcastData = new float[totalElements];

            Parallel.For(0, totalElements, linearIndex =>
            {
                int originalIndex = 0;
                int remaining = linearIndex;

                for (int dim = 0; dim < targetRank; dim++)
                {
                    int targetSize = targetShape[dim];
                    int tensorSize = paddedTensorShape[dim];

                    int coord = remaining / targetStrides[dim];
                    remaining %= targetStrides[dim];

                    if (tensorSize != 1)
                    {
                        coord %= tensorSize;
                        originalIndex += coord * tensorStrides[dim];
                    }
                }

                broadcastData[linearIndex] = tensor.Data[originalIndex];
            });

            bool requiresGrad = tensor.RequiresGrad;
            Tensor result = new Tensor(broadcastData, targetShape, requiresGrad);

            if (buildGraph && requiresGrad)
            {
                result.GradFn = GradFunction.BroadcastGradFn;
                result.OpArgs["Broadcast_targetShape"] = targetShape;
                result.OpArgs["Broadcast_originalShape"] = tensor.Shape;
                result.LeftLeaf = tensor;
                result.LeftLeaf.Father = result;
            }

            return result;
        }

        public static (Tensor, Tensor) Broadcast(Tensor a, Tensor b, bool buildGraph = true)
        {
            int[] broadcastShape = GetBroadcastShape(a.Shape, b.Shape);
            Tensor aBroadcast = BroadcastTo(a, broadcastShape, buildGraph);
            Tensor bBroadcast = BroadcastTo(b, broadcastShape, buildGraph);

            return (aBroadcast, bBroadcast);
        }
    }
}
