using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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

            Tensor result = new Tensor(Operator.Add(aBroadcast.Data, bBroadcast.Data), aBroadcast.Shape, aBroadcast.RequiresGrad || bBroadcast.RequiresGrad);

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

            Tensor result = new Tensor(Operator.Subtract(aBroadcast.Data, bBroadcast.Data), aBroadcast.Shape, aBroadcast.RequiresGrad || bBroadcast.RequiresGrad);

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

            Tensor result = new Tensor(Operator.Multiply(aBroadcast.Data, bBroadcast.Data), aBroadcast.Shape, aBroadcast.RequiresGrad || bBroadcast.RequiresGrad);

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

            Tensor result = new Tensor(Operator.Divide(aBroadcast.Data, bBroadcast.Data), aBroadcast.Shape, aBroadcast.RequiresGrad || bBroadcast.RequiresGrad);

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

            Tensor result = new Tensor(Operator.Negate(a.Data), a.Shape, a.RequiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.NegGradFn;
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

           

            Tensor result = new Tensor(Operator.MatMul(a.Data, b.Data, aRows, aCols, bCols), [aRows, bCols], a.RequiresGrad || b.RequiresGrad);

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

        public static Tensor Transpose(Tensor tensor, bool buildGraph = true)
        {
            if (tensor.Rank != 2)
                throw new InvalidOperationException("Transpose requires a 2D tensor.");

            int[] originalShape = tensor.Shape;
            int rows = originalShape[0];
            int cols = originalShape[1];

            int[] newShape = [cols, rows];
            float[] newData = new float[rows * cols];

            Parallel.For(0, rows, i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    int sourceIndex = i * cols + j;
                    int destIndex = j * rows + i;
                    newData[destIndex] = tensor.Data[sourceIndex];
                }
            });

            Tensor result = new Tensor(newData, newShape, tensor.RequiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.TransposeGradFn;
                result.LeftLeaf = tensor;
                result.LeftLeaf.Father = result;
            }

            return result;
        }

        public static Tensor Repeat(Tensor tensor, int axis, int repeat = 2)
        {
            if (tensor == null) 
                throw new ArgumentNullException(nameof(tensor));
            if (axis < 0 || axis >= tensor.Rank)
                throw new ArgumentOutOfRangeException(nameof(axis), "Axis must be within the tensor's rank.");
            if (repeat < 1)
                throw new ArgumentOutOfRangeException(nameof(repeat), "Repeat must be at least 1.");

            int[] newShape = tensor.Shape.ToArray();
            newShape[axis] *= repeat;

            int outer = 1;
            for (int i = 0; i < axis; i++)
            {
                outer *= tensor.Shape[i];
            }

            int dim = tensor.Shape[axis];
            int inner = 1;
            for (int i = axis + 1; i < tensor.Rank; i++)
            {
                inner *= tensor.Shape[i];
            }

            float[] newData = new float[outer * dim * repeat * inner];

            for (int o = 0; o < outer; o++)
            {
                int sourceOuterOffset = o * dim * inner;
                int targetOuterOffset = o * dim * repeat * inner;

                for (int d = 0; d < dim; d++)
                {
                    int sourceStart = sourceOuterOffset + d * inner;
                    int targetStartBase = targetOuterOffset + d * repeat * inner;

                    for (int r = 0; r < repeat; r++)
                    {
                        int targetStart = targetStartBase + r * inner;
                        Array.Copy(tensor.Data, sourceStart, newData, targetStart, inner);
                    }
                }
            }

            Tensor result = new Tensor(newData, newShape, tensor.RequiresGrad);

            return result;
        }

        public static Tensor Reshape(Tensor tensor, int[] newShape)
        {
            int totalElements = tensor.Data.Length;
            int newTotal = newShape.Aggregate(1, (a, b) => a * b);

            if (totalElements != newTotal)
                throw new ArgumentException($"Cannot reshape tensor from [{string.Join(", ", tensor.Shape)}] to [{string.Join(", ", newShape)}]");

            Tensor result = new Tensor(tensor.Data, newShape, tensor.RequiresGrad);

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
