using System.Runtime.InteropServices;

namespace VerbNet.Core
{
    public class Tensor : IDisposable
    {
        public AlignedArray<float> Data;
        public int Length => Data.Length;
        public int[] Shape;
        public int Rank => Shape.Length;
        public string Name;

        public bool RequiresGrad;
        public Tensor Gradient;
        public Func<Tensor, Tensor, Tensor, Dictionary<string, object>, (Tensor, Tensor)> GradFn;
        public Dictionary<string, object> OpArgs;
        public Tensor Father;
        public Tensor LeftLeaf;
        public Tensor RightLeaf;

        public readonly static Tensor Zero = Tensor.Create(0f, false, "Zero");
        public readonly static Tensor One = Tensor.Create(1f, false, "One");

        private static ParallelOptions _parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = 2 };

        private bool disposed;

        public Tensor(int[] shape, bool requiresGrad = false, string name = "")
        {
            if (shape == null)
                throw new ArgumentNullException(nameof(shape), "Shape cannot be null");
            if (shape.Length == 0)
                throw new ArgumentException("Shape must have at least one dimension");
            foreach (int dim in shape)
            {
                if (dim <= 0)
                    throw new ArgumentException("All dimensions must be positive integers");
            }

            Data = new AlignedArray<float>(shape.Aggregate(1, (a, b) => a * b), 32);
            Shape = (int[])shape.Clone();
            Name = name;

            RequiresGrad = requiresGrad;
            GradFn = null;
            OpArgs = new Dictionary<string, object>();
            Father = null;
            LeftLeaf = null;
            RightLeaf = null;
            if (requiresGrad)
            {
                Gradient = new Tensor(Shape, false);
            }
        }

        public Tensor(float[] data, int[] shape, bool requiresGrad = false, string name = "")
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data), "Data cannot be null");
            if (shape == null)
                throw new ArgumentNullException(nameof(shape), "Shape cannot be null");
            if (shape.Length == 0)
                throw new ArgumentException("Shape must have at least one dimension");
            foreach (int dim in shape)
            {
                if (dim <= 0)
                    throw new ArgumentException("All dimensions must be positive integers");
            }

            int totalElements = shape.Aggregate(1, (a, b) => a * b);
            if (data.Length != totalElements)
                throw new ArgumentException($"Data length ({data.Length}) does not match shape product ({totalElements})");

            Data = new AlignedArray<float>(data, 32);
            Shape = (int[])shape.Clone();
            Name = name;

            RequiresGrad = requiresGrad;
            GradFn = null;
            OpArgs = new Dictionary<string, object>();
            Father = null;
            LeftLeaf = null;
            RightLeaf = null;
            if (requiresGrad)
            {
                Gradient = new Tensor(Shape, false);
            }
        }

        public Tensor(AlignedArray<float> data, int[] shape, bool requiresGrad = false, string name = "")
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data), "Data cannot be null");
            if (shape == null)
                throw new ArgumentNullException(nameof(shape), "Shape cannot be null");
            if (shape.Length == 0)
                throw new ArgumentException("Shape must have at least one dimension");
            foreach (int dim in shape)
            {
                if (dim <= 0)
                    throw new ArgumentException("All dimensions must be positive integers");
            }

            int totalElements = shape.Aggregate(1, (a, b) => a * b);
            if (data.Length != totalElements)
                throw new ArgumentException($"Data length ({data.Length}) does not match shape product ({totalElements})");

            Data = data;
            Shape = (int[])shape.Clone();
            Name = name;

            RequiresGrad = requiresGrad;
            GradFn = null;
            OpArgs = new Dictionary<string, object>();
            Father = null;
            LeftLeaf = null;
            RightLeaf = null;
            if (requiresGrad)
            {
                Gradient = new Tensor(Shape, false);
            }
        }

        private Tensor()
        {

        }

        public float this[params int[] indices]
        {
            get
            {
                if (indices.Length != Rank)
                {
                    throw new ArgumentException($"Index count must equal the tensor's rank {Rank}");
                }

                int index = 0;
                int stride = 1;
                for (int i = Rank - 1; i >= 0; i--)
                {
                    if (indices[i] < 0 || indices[i] >= Shape[i])
                    {
                        throw new ArgumentOutOfRangeException($"Index {i} out of range, maximum value should be {Shape[i] - 1}");
                    }
                    index += indices[i] * stride;
                    stride *= Shape[i];
                }

                return Data[index];
            }
            set
            {
                if (indices.Length != Rank)
                {
                    throw new ArgumentException($"Index count must equal the tensor's rank {Rank}");
                }

                int index = 0;
                int stride = 1;
                for (int i = Rank - 1; i >= 0; i--)
                {
                    if (indices[i] < 0 || indices[i] >= Shape[i])
                    {
                        throw new ArgumentOutOfRangeException($"Index {i} out of range, maximum value should be {Shape[i] - 1}");
                    }
                    index += indices[i] * stride;
                    stride *= Shape[i];
                }

                Data[index] = value;
            }
        }

        public static Tensor Create(float value, bool requiresGrad = false, string name = "")
        {
            return new Tensor([value], [1], requiresGrad, name);
        }

        public static Tensor operator +(Tensor a, Tensor b) => TensorOperator.Add(a, b, true, true);
        public static Tensor operator +(Tensor a, float b) => TensorOperator.Add(a, b);
        public static Tensor operator +(float a, Tensor b) => TensorOperator.Add(b, a);
        public static Tensor operator -(Tensor a, Tensor b) => TensorOperator.Subtract(a, b, true, true);
        public static Tensor operator -(Tensor a, float b) => TensorOperator.Subtract(a, b);
        public static Tensor operator -(float a, Tensor b) => TensorOperator.Subtract(a, b);
        public static Tensor operator *(Tensor a, Tensor b) => TensorOperator.Multiply(a, b, true, true);
        public static Tensor operator *(Tensor a, float b) => TensorOperator.Multiply(a, b);
        public static Tensor operator *(float a, Tensor b) => TensorOperator.Multiply(b, a);
        public static Tensor operator /(Tensor a, Tensor b) => TensorOperator.Divide(a, b, true, true);
        public static Tensor operator /(Tensor a, float b) => TensorOperator.Divide(a, b);
        public static Tensor operator /(float a, Tensor b) => TensorOperator.Divide(a, b);
        public static Tensor operator -(Tensor a) => TensorOperator.Negate(a, true, true);

        public Tensor Repeat(int axis, int repeat = 2) => TensorOperator.Repeat(this, axis, repeat, true, true);
        public static Tensor Repeat(Tensor a, int axis, int repeat = 2) => TensorOperator.Repeat(a, axis, repeat, true, true);
        public Tensor Reshape(int[] shape) => TensorOperator.Reshape(this, shape, true, true);
        public static Tensor Reshape(Tensor a, int[] shape) => TensorOperator.Reshape(a, shape, true, true);
        public Tensor Cat(Tensor b, int dim) => TensorOperator.Concat(this, b, dim, true, true);
        public static Tensor Cat(Tensor a, Tensor b, int dim) => TensorOperator.Concat(a, b, dim, true, true);
        public Tensor Sum(int dim = -1, bool keepDim = false) => TensorOperator.Sum(this, dim, keepDim, true, true);
        public static Tensor Sum(Tensor a, int dim = -1, bool keepDim = false) => TensorOperator.Sum(a, dim, keepDim, true, true);
        public Tensor Mean(int dim = -1, bool keepDim = false) => TensorOperator.Mean(this, dim, keepDim, true, true);
        public static Tensor Mean(Tensor a, int dim = -1, bool keepDim = false) => TensorOperator.Mean(a, dim, keepDim, true, true);
        public Tensor Variance(int dim = -1, bool keepDim = false) => TensorOperator.Variance(this, dim, keepDim, true, true);
        public static Tensor Variance(Tensor a, int dim = -1, bool keepDim = false) => TensorOperator.Variance(a, dim, keepDim, true, true);

        public static Tensor Abs(Tensor a) => TensorOperator.Abs(a, true, true);
        public static Tensor Sign(Tensor a) => TensorOperator.Sign(a, true, true);
        public static Tensor Sqrt(Tensor a) => TensorOperator.Sqrt(a, true, true);
        public static Tensor Log(Tensor a) => TensorOperator.LogE(a, true, true);
        public static Tensor Exp(Tensor a) => TensorOperator.Exp(a, true, true);
        public static Tensor Pow(Tensor a, float b) => TensorOperator.Power(a, b, true, true);
        public static Tensor Sin(Tensor a) => TensorOperator.Sin(a, true, true);
        public static Tensor Cos(Tensor a) => TensorOperator.Cos(a, true, true);
        public static Tensor Tan(Tensor a) => TensorOperator.Tan(a, true, true);
        public static Tensor Sinh(Tensor a) => TensorOperator.Sinh(a, true, true);
        public static Tensor Cosh(Tensor a) => TensorOperator.Cosh(a, true, true);
        public static Tensor Tanh(Tensor a) => TensorOperator.Tanh(a, true, true);
        public static Tensor Max(Tensor a, Tensor b) => TensorOperator.Max(a, b, true, true);
        public static Tensor Max(Tensor a, float b) => TensorOperator.Max(a, b);
        public static Tensor Max(float a, Tensor b) => TensorOperator.Max(a, b);
        public static Tensor Min(Tensor a, Tensor b) => TensorOperator.Min(a, b, true, true);
        public static Tensor Min(Tensor a, float b) => TensorOperator.Min(a, b);
        public static Tensor Min(float a, Tensor b) => TensorOperator.Min(a, b);
        public static Tensor Floor(Tensor a) => TensorOperator.Floor(a, true, true);
        public static Tensor Ceiling(Tensor a) => TensorOperator.Ceiling(a, true, true);
        public static Tensor Round(Tensor a) => TensorOperator.Round(a, true, true);
        public static Tensor Transpose(Tensor a) => TensorOperator.Transpose(a, true, true);
        public static Tensor MatMul(Tensor a, Tensor b) => TensorOperator.MatMul(a, b, true, true);

        public static Tensor Random(int[] shape, float scale = 1f, bool requiresGrad = false, string name = "") => TensorOperator.Random(shape, scale, requiresGrad, name);
        public static Tensor Ones(int[] shape, bool requiresGrad = false, string name = "") => TensorOperator.Ones(shape, requiresGrad, name);

        public void Backward(Tensor externalGradient = null)
        {
            if (!RequiresGrad)
                throw new InvalidOperationException("Cannot call Backward on a tensor that does not require gradients.");

            if (externalGradient == null)
            {
                externalGradient = One;
            }

            lock (this)
            {
                Gradient = TensorOperator.Add(Gradient, externalGradient, false, false);
            }

            if (GradFn != null)
            {
                (Tensor leftGrad, Tensor rightGrad) = GradFn(Gradient, LeftLeaf, RightLeaf, OpArgs);

                Parallel.Invoke(_parallelOptions,
                    () =>
                    {
                        if (LeftLeaf != null && LeftLeaf.RequiresGrad)
                        {
                            if (leftGrad == null)
                                throw new InvalidOperationException("Left gradient is null for a tensor that requires grad.");
                            LeftLeaf.Backward(leftGrad);
                        }
                    },
                    () =>
                    {
                        if (RightLeaf != null && RightLeaf.RequiresGrad)
                        {
                            if (rightGrad == null)
                                throw new InvalidOperationException("Right gradient is null for a tensor that requires grad.");
                            RightLeaf.Backward(rightGrad);
                        }
                    }
                );
            }
        }

        public Tensor Clone()
        {
            Tensor copy = new Tensor()
            {
                Data = Data.Clone(),
                Shape = (int[])Shape.Clone(),
                Name = Name,
                RequiresGrad = RequiresGrad,
                Gradient = Gradient?.Clone(),
                GradFn = null,
                OpArgs = new Dictionary<string, object>(),
                Father = null,
                LeftLeaf = null,
                RightLeaf = null
            };

            return copy;
        }

        public BinaryWriter Write(BinaryWriter bw)
        {
            bw.Write(Name);

            bw.Write(Shape.Length);
            bw.Write(MemoryMarshal.AsBytes(Shape.AsSpan()));

            bw.Write(Data.Length);
            bw.Write(MemoryMarshal.AsBytes(Data.AsSpan()));

            bw.Write(RequiresGrad);
            if (RequiresGrad)
            {
                bw = Gradient.Write(bw);
            }

            return bw;
        }

        public unsafe BinaryReader Read(BinaryReader br)
        {
            Name = br.ReadString();

            int shapeLength = br.ReadInt32();
            Shape = new int[shapeLength];
            byte[] shapeBytes = br.ReadBytes(shapeLength * sizeof(int));
            Buffer.BlockCopy(shapeBytes, 0, Shape, 0, shapeBytes.Length);

            int dataLength = br.ReadInt32();
            Data = new AlignedArray<float>(dataLength);
            byte[] dataBytes = br.ReadBytes(dataLength * sizeof(float));

            fixed (byte* sourcePtr = dataBytes)
            fixed (float* destPtr = Data.AsSpan())
            {
                Buffer.MemoryCopy(sourcePtr, destPtr, dataBytes.Length, dataBytes.Length);
            }

            RequiresGrad = br.ReadBoolean();
            if (RequiresGrad)
            {
                Gradient = new Tensor();
                Gradient.Read(br);
            }

            return br;
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposed)
            {
                Data.Dispose();
                Gradient?.Dispose();

                disposed = true;
            }
        }

        ~Tensor()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);

            Father = null;
            LeftLeaf = null;
            RightLeaf = null;

            GC.SuppressFinalize(this);
        }
    }
}
