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
            (aBroadcast, bBroadcast) = Broadcast(a, b, buildGraph);

            bool requiresGrad = buildGraph ? (a.RequiresGrad || b.RequiresGrad) : false;
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
            (aBroadcast, bBroadcast) = Broadcast(a, b, buildGraph);

            bool requiresGrad = buildGraph ? (a.RequiresGrad || b.RequiresGrad) : false;
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
            (aBroadcast, bBroadcast) = Broadcast(a, b, buildGraph);

            bool requiresGrad = buildGraph ? (a.RequiresGrad || b.RequiresGrad) : false;
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
            (aBroadcast, bBroadcast) = Broadcast(a, b, buildGraph);

            bool requiresGrad = buildGraph ? (a.RequiresGrad || b.RequiresGrad) : false;
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

            bool requiresGrad = buildGraph ? a.RequiresGrad : false;
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

            bool requiresGrad = buildGraph ? a.RequiresGrad : false;
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

            bool requiresGrad = buildGraph ? a.RequiresGrad : false;
            Tensor result = new Tensor(Operator.Sign(a.Data), a.Shape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.SignGradFn;
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

            bool requiresGrad = buildGraph ? (a.RequiresGrad || b.RequiresGrad) : false;
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

            bool requiresGrad = buildGraph ? a.RequiresGrad : false;
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

            int[] newShape = a.Shape.ToArray();
            newShape[axis] *= repeat;

            if (repeat == 1)
            {
                return Reshape(a, newShape);
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

            Tensor result = new Tensor(newData, newShape, a.RequiresGrad);

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

        public static Tensor Reshape(Tensor tensor, int[] newShape, bool buildGraph = true)
        {
            int length = tensor.Data.Length;
            int newLength = newShape.Aggregate(1, (a, b) => a * b);

            if (length != newLength)
                throw new ArgumentException($"Cannot reshape tensor from [{string.Join(", ", tensor.Shape)}] to [{string.Join(", ", newShape)}]");

            Tensor result = new Tensor(tensor.Data, newShape, tensor.RequiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.ReshapeGradFn;
                result.LeftLeaf = tensor;
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

        public static Tensor BroadcastTo(Tensor tensor, int[] targetShape)
        {
            int[] broadcastedShape = GetBroadcastShape(tensor.Shape, targetShape);

            if (!Enumerable.SequenceEqual(broadcastedShape, targetShape))
                throw new ArgumentException($"Cannot broadcast tensor from shape [{string.Join(", ", tensor.Shape)}] to [{string.Join(", ", targetShape)}]");

            int targetRank = targetShape.Length;
            int tensorRank = tensor.Shape.Length;
            int padLength = targetRank - tensorRank;

            int[] paddedShape = new int[targetRank];
            for (int i = 0; i < targetRank; i++)
            {
                paddedShape[i] = (i < padLength) ? 1 : tensor.Shape[i - padLength];
            }

            Tensor current = Reshape(tensor, paddedShape);

            for (int i = 0; i < targetRank; i++)
            {
                if (current.Shape[i] == 1 && targetShape[i] > 1)
                {
                    current = Repeat(current, i, targetShape[i]);
                }
            }

            return current;
        }

        public static (Tensor, Tensor) Broadcast(Tensor a, Tensor b, bool buildGraph)
        {
            int[] broadcastShape = GetBroadcastShape(a.Shape, b.Shape);
            Tensor aBroadcast = BroadcastTo(a, broadcastShape);
            Tensor bBroadcast = BroadcastTo(b, broadcastShape);

            return (aBroadcast, bBroadcast);
        }
    }
}
