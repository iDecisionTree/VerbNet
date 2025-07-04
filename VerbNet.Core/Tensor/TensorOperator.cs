﻿namespace VerbNet.Core
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
            (aBroadcast, bBroadcast) = Broadcast(a, b, false, false);

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

        public static Tensor Add(Tensor a, float b)
        {
            Tensor bTensor = Tensor.Create(b);

            return Add(a, bTensor, true, true);
        }

        public static Tensor Subtract(Tensor a, Tensor b, bool buildGraph = true, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));
            if (b == null)
                throw new ArgumentNullException(nameof(b));

            Tensor aBroadcast, bBroadcast;
            (aBroadcast, bBroadcast) = Broadcast(a, b, false, false);

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

            return Subtract(a, bTensor, true, true);
        }

        public static Tensor Subtract(float a, Tensor b)
        {
            Tensor aTensor = Tensor.Create(a);

            return Subtract(aTensor, b, true, true);
        }

        public static Tensor Multiply(Tensor a, Tensor b, bool buildGraph = true, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));
            if (b == null)
                throw new ArgumentNullException(nameof(b));

            Tensor aBroadcast, bBroadcast;
            (aBroadcast, bBroadcast) = Broadcast(a, b, false, false);

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

            return Multiply(a, bTensor, true, true);
        }

        public static Tensor Divide(Tensor a, Tensor b, bool buildGraph = true, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));
            if (b == null)
                throw new ArgumentNullException(nameof(b));

            Tensor aBroadcast, bBroadcast;
            (aBroadcast, bBroadcast) = Broadcast(a, b, false, false);

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

            return Divide(a, bTensor, true, true);
        }

        public static Tensor Divide(float a, Tensor b)
        {
            Tensor aTensor = Tensor.Create(a);

            return Divide(aTensor, b, true, true);
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
            (aBroadcast, bBroadcast) = Broadcast(a, b, false, false);

            bool requiresGrad = computeGrad && (a.RequiresGrad || b.RequiresGrad);
            Tensor result = new Tensor(Operator.Max(aBroadcast.Data, bBroadcast.Data), aBroadcast.Shape, requiresGrad);

            if (buildGraph)
            {
                bool[] maskA = new bool[aBroadcast.Length];
                bool[] maskB = new bool[bBroadcast.Length];
                bool[] equalMask = new bool[aBroadcast.Length];

                for (int i = 0; i < aBroadcast.Length; i++)
                {
                    if (aBroadcast.Data[i] > bBroadcast.Data[i])
                    {
                        maskA[i] = true;
                    }
                    else if (aBroadcast.Data[i] < bBroadcast.Data[i])
                    {
                        maskB[i] = true;
                    }
                    else
                    {
                        equalMask[i] = true;
                    }
                }

                result.GradFn = GradFunction.MaxGradFn;
                result.OpArgs.Add("Max_maskA", maskA);
                result.OpArgs.Add("Max_maskB", maskB);
                result.OpArgs.Add("Max_equalMask", equalMask);
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
                result.RightLeaf = b;
                result.RightLeaf.Father = result;
            }
            else if (requiresGrad && computeGrad)
            {
                if (a.RequiresGrad && b.RequiresGrad)
                {
                    result.Gradient = Max(a.Gradient, b.Gradient, false, false);
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

        public static Tensor Max(Tensor a, float b)
        {
            Tensor bTensor = Tensor.Create(b);

            return Max(a, bTensor, true, true);
        }

        public static Tensor Max(float a, Tensor b)
        {
            Tensor aTensor = Tensor.Create(a);

            return Max(aTensor, b, true, true);
        }

        public static Tensor Min(Tensor a, Tensor b, bool buildGraph = true, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));
            if (b == null)
                throw new ArgumentNullException(nameof(b));

            Tensor aBroadcast, bBroadcast;
            (aBroadcast, bBroadcast) = Broadcast(a, b, false, false);

            bool requiresGrad = computeGrad && (a.RequiresGrad || b.RequiresGrad);
            Tensor result = new Tensor(Operator.Min(aBroadcast.Data, bBroadcast.Data), aBroadcast.Shape, requiresGrad);

            if (buildGraph)
            {
                bool[] maskA = new bool[aBroadcast.Length];
                bool[] maskB = new bool[bBroadcast.Length];
                bool[] equalMask = new bool[aBroadcast.Length];

                for (int i = 0; i < aBroadcast.Length; i++)
                {
                    if (aBroadcast.Data[i] > bBroadcast.Data[i])
                    {
                        maskA[i] = true;
                    }
                    else if (aBroadcast.Data[i] < bBroadcast.Data[i])
                    {
                        maskB[i] = true;
                    }
                    else
                    {
                        equalMask[i] = true;
                    }
                }

                result.GradFn = GradFunction.MinGradFn;
                result.OpArgs.Add("Min_maskA", maskA);
                result.OpArgs.Add("Min_maskB", maskB);
                result.OpArgs.Add("Min_equalMask", equalMask);
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
                result.RightLeaf = b;
                result.RightLeaf.Father = result;
            }
            else if (requiresGrad && computeGrad)
            {
                if (a.RequiresGrad && b.RequiresGrad)
                {
                    result.Gradient = Min(a.Gradient, b.Gradient, false, false);
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

        public static Tensor Min(Tensor a, float b)
        {
            Tensor bTensor = Tensor.Create(b);

            return Min(a, bTensor, true, true);
        }

        public static Tensor Min(float a, Tensor b)
        {
            Tensor aTensor = Tensor.Create(a);

            return Min(aTensor, b, true, true);
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

        public static Tensor Repeat(Tensor a, int axis, int repeat = 2, bool buildGraph = true, bool computeGrad = true)
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

            if (buildGraph)
            {
                result.GradFn = GradFunction.RepeatGradFn;
                result.OpArgs.Add("Repeat_axis", axis);
                result.OpArgs.Add("Repeat_repeat", repeat);
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }
            else if (requiresGrad && computeGrad)
            {
                result.Gradient = Repeat(a.Gradient, axis, repeat, false, false);
            }

            return result;
        }

        public static Tensor Reshape(Tensor a, int[] newShape, bool buildGraph = true, bool computeGrad = true)
        {
            int length = a.Data.Length;
            int newLength = newShape.Aggregate(1, (a, b) => a * b);

            if (length != newLength)
                throw new ArgumentException($"Cannot reshape tensor from [{string.Join(", ", a.Shape)}] to [{string.Join(", ", newShape)}]");

            bool requiresGrad = computeGrad && a.RequiresGrad;
            Tensor result = new Tensor(a.Data, newShape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.ReshapeGradFn;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }
            else if (requiresGrad && computeGrad)
            {
                result.Gradient = Reshape(a.Gradient, newShape, false, false);
            }

            return result;
        }

        public static Tensor Concat(Tensor a, Tensor b, int dim, bool buildGraph = true, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));
            if (b == null)
                throw new ArgumentNullException(nameof(b));
            if (dim < 0 || dim >= a.Rank || dim >= b.Rank)
                throw new ArgumentOutOfRangeException(nameof(dim), "Dimension must be within the rank of both tensors.");
            if (a.Rank != b.Rank)
                throw new ArgumentException("Both tensors must have the same rank for concatenation.");

            for (int i = 0; i < a.Rank; i++)
            {
                if (i != dim && a.Shape[i] != b.Shape[i])
                {
                    throw new ArgumentException($"Tensors cannot be concatenated along dimension {dim} because their shapes differ in dimension {i}: {a.Shape[i]} vs {b.Shape[i]}");
                }
            }

            int[] newShape = (int[])a.Shape.Clone();
            newShape[dim] = a.Shape[dim] + b.Shape[dim];

            int totalElements = newShape.Aggregate(1, (acc, val) => acc * val);
            AlignedArray<float> resultData = new AlignedArray<float>(totalElements, a.Data.Alignment);

            Parallel.For(0, totalElements, i =>
            {
                int[] resultIndices = GetMultiIndex(i, newShape);
                int dimValue = resultIndices[dim];
                if (dimValue < a.Shape[dim])
                {
                    int[] aIndices = (int[])resultIndices.Clone();
                    int aLinearIndex = GetLinearIndex(aIndices, a.Shape);
                    resultData[i] = a.Data[aLinearIndex];
                }
                else
                {
                    int[] bIndices = (int[])resultIndices.Clone();
                    bIndices[dim] -= a.Shape[dim];
                    int bLinearIndex = GetLinearIndex(bIndices, b.Shape);
                    resultData[i] = b.Data[bLinearIndex];
                }
            });

            bool requiresGrad = computeGrad && (a.RequiresGrad || b.RequiresGrad);
            Tensor result = new Tensor(resultData, newShape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.ConcatGradFn;
                result.OpArgs["Concat_dim"] = dim;
                result.LeftLeaf = a;
                result.RightLeaf = b;
                result.LeftLeaf.Father = result;
                result.RightLeaf.Father = result;
            }
            else if (requiresGrad && computeGrad)
            {
                result.Gradient = Concat(a.Gradient, b.Gradient, dim, false, false);
            }

            return result;
        }

        public static Tensor Sum(Tensor a, int dim = -1, bool keepDim = false, bool buildGraph = true, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            if (dim >= a.Rank)
                throw new ArgumentOutOfRangeException(nameof(dim), "维度必须处于张量的阶数范围内");

            int[] newShape;
            AlignedArray<float> resultData;

            if (dim != -1)
            {
                int axis = dim;

                int outer = 1;
                for (int i = 0; i < axis; i++)
                {
                    outer *= a.Shape[i];
                }

                int dimSize = a.Shape[axis];

                int inner = 1;
                for (int i = axis + 1; i < a.Rank; i++)
                {
                    inner *= a.Shape[i];
                }

                if (keepDim)
                {
                    newShape = a.Shape;
                    newShape[axis] = 1;
                }
                else
                {
                    newShape = new int[a.Rank - 1];
                    for (int i = 0, j = 0; i < a.Rank; i++)
                    {
                        if (i != axis) newShape[j++] = a.Shape[i];
                    }
                }

                resultData = new AlignedArray<float>(outer * inner, a.Data.Alignment);

                Parallel.For(0, outer, o =>
                {
                    for (int i = 0; i < inner; i++)
                    {
                        float sum = 0f;
                        for (int d = 0; d < dimSize; d++)
                        {
                            int index = o * dimSize * inner + d * inner + i;
                            sum += a.Data[index];
                        }
                        resultData[o * inner + i] = sum;
                    }
                });
            }
            else
            {
                newShape = keepDim ? Enumerable.Repeat(1, a.Rank).ToArray() : [1];

                float sum = 0f;
                for (int i = 0; i < a.Data.Length; i++)
                {
                    sum += a.Data[i];
                }

                resultData = new AlignedArray<float>(1, a.Data.Alignment);
                resultData[0] = sum;
            }

            bool requiresGrad = computeGrad && a.RequiresGrad;
            Tensor result = new Tensor(resultData, newShape, requiresGrad);

            if (buildGraph)
            {
                result.GradFn = GradFunction.SumGradFn;
                result.OpArgs["Sum_dim"] = dim;
                result.OpArgs["Sum_keepDim"] = keepDim;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }
            else if (requiresGrad && computeGrad)
            {
                if (dim == -1)
                {
                    Tensor ones = Ones(a.Shape);
                    result.Gradient = keepDim ? ones : Sum(ones, dim, false, false, false);
                }
                else
                {
                    result.Gradient = Ones(a.Shape);
                }
            }

            return result;
        }

        public static Tensor Mean(Tensor a, int dim = -1, bool keepDim = false, bool buildGraph = true, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            Tensor sum = Sum(a, dim, keepDim, true, true);

            int numElements;
            if (dim == -1)
            {
                numElements = a.Length;
            }
            else
            {
                if (dim < 0 || dim >= a.Rank)
                    throw new ArgumentOutOfRangeException(nameof(dim), "Dimension must be within the tensor's rank");
                numElements = a.Shape[dim];
            }

            Tensor result = Divide(sum, numElements);

            if (buildGraph)
            {
                result.GradFn = GradFunction.MeanGradFn;
                result.OpArgs["Mean_dim"] = dim;
                result.OpArgs["Mean_keepDim"] = keepDim;
                result.OpArgs["Mean_numElements"] = numElements;
                result.LeftLeaf = a;
                result.LeftLeaf.Father = result;
            }
            else if (computeGrad && a.RequiresGrad)
            {
                if (dim == -1)
                {
                    Tensor ones = Ones(a.Shape);
                    result.Gradient = Divide(ones, numElements);
                }
                else
                {
                    result.Gradient = Divide(Ones(a.Shape), numElements);
                }
            }

            return result;
        }

        public static Tensor Variance(Tensor a, int dim = -1, bool keepDim = false, bool buildGraph = true, bool computeGrad = true)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));

            Tensor mean = Mean(a, dim, true, true, true);

            Tensor squaredDiff = Power(Subtract(a, mean), 2, true, true);

            Tensor variance = Mean(squaredDiff, dim, keepDim, buildGraph, computeGrad);

            if (buildGraph)
            {
                variance.GradFn = GradFunction.VarianceGradFn;
                variance.OpArgs["Variance_dim"] = dim;
                variance.OpArgs["Variance_keepDim"] = keepDim;
                variance.LeftLeaf = a;
                variance.LeftLeaf.Father = variance;
            }
            else if (computeGrad && a.RequiresGrad)
            {
                int n = (dim == -1) ? a.Length : a.Shape[dim];
                Tensor diff = Subtract(a, mean);
                variance.Gradient = Multiply(Divide(diff, n - 1), Tensor.Create(2f));
            }

            return variance;
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
                data[i] = (float)(rand.NextDouble() * 2d - 1d) * scale;
            }

            Tensor result = new Tensor(data, shape, requiresGrad, name);

            return result;
        }

        public static Tensor Ones(int[] shape, bool requireGrad = false, string name = "")
        {
            Tensor result = new Tensor(shape, requireGrad, name);
            result.Data.Fill(1f);

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
                    throw new InvalidOperationException($"Shapes [{string.Join(",", aShape)}] and [{string.Join(",", bShape)}] cannot be broadcasted");
                }
            }

            reversedResult.Reverse();
            return reversedResult.ToArray();
        }

        public static Tensor BroadcastTo(Tensor tensor, int[] targetShape, bool buildGraph = true, bool computeGrad = true)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));
            if (targetShape == null)
                throw new ArgumentNullException(nameof(targetShape));

            int[] broadcastedShape = GetBroadcastShape(tensor.Shape, targetShape);
            if (!Enumerable.SequenceEqual(broadcastedShape, targetShape))
                throw new ArgumentException($"Cannot broadcast tensor from shape [{string.Join(", ", tensor.Shape)}] to [{string.Join(", ", targetShape)}]");

            if (tensor.Shape.SequenceEqual(targetShape))
                return buildGraph ? tensor.Clone() : tensor;

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

            if (buildGraph)
            {
                result.GradFn = GradFunction.BroadcastGradFn;
                result.OpArgs["Broadcast_targetShape"] = targetShape;
                result.OpArgs["Broadcast_originalShape"] = tensor.Shape;
                result.LeftLeaf = tensor;
                result.LeftLeaf.Father = result;
            }
            else if (requiresGrad && computeGrad)
            {
                result.Gradient = BroadcastTo(tensor.Gradient, targetShape, false, false);
            }

            return result;
        }

        public static (Tensor, Tensor) Broadcast(Tensor a, Tensor b, bool buildGraph = true, bool computeGrad = true)
        {
            int[] broadcastShape = GetBroadcastShape(a.Shape, b.Shape);
            Tensor aBroadcast = BroadcastTo(a, broadcastShape, buildGraph, computeGrad);
            Tensor bBroadcast = BroadcastTo(b, broadcastShape, buildGraph, computeGrad);

            return (aBroadcast, bBroadcast);
        }

        public static int[] GetMultiIndex(int linearIndex, int[] shape)
        {
            int[] indices = new int[shape.Length];
            int temp = linearIndex;
            for (int i = shape.Length - 1; i >= 0; i--)
            {
                indices[i] = temp % shape[i];
                temp /= shape[i];
            }

            return indices;
        }

        public static int GetLinearIndex(int[] indices, int[] shape)
        {
            int index = 0;
            int stride = 1;
            for (int i = shape.Length - 1; i >= 0; i--)
            {
                index += indices[i] * stride;
                stride *= shape[i];
            }

            return index;
        }

        public static int[] BroadcastedToOriginalIndex(int[] index, int[] originalShape, int[] broadcastedShape)
        {
            if (index.Length != broadcastedShape.Length)
                throw new ArgumentException("Index and BroadcastedShape must have the same length.", nameof(index));

            int maxRank = Math.Max(originalShape.Length, broadcastedShape.Length);

            int[] expandedOriginal = new int[maxRank];
            int[] expandedBroadcasted = new int[maxRank];

            Array.Copy(originalShape, 0, expandedOriginal, maxRank - originalShape.Length, originalShape.Length);
            Array.Copy(broadcastedShape, 0, expandedBroadcasted, maxRank - broadcastedShape.Length, broadcastedShape.Length);

            int[] expandedIndex = new int[maxRank];
            Array.Copy(index, 0, expandedIndex, maxRank - index.Length, index.Length);

            for (int i = 0; i < maxRank; i++)
            {
                if (expandedOriginal[i] == 1 && expandedBroadcasted[i] != 1)
                {
                    expandedIndex[i] = 0;
                }
            }

            int[] result = new int[originalShape.Length];
            Array.Copy(expandedIndex, maxRank - originalShape.Length, result, 0, originalShape.Length);

            return result;
        }
    }
}