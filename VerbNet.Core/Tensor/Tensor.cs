namespace VerbNet.Core
{
    public class Tensor
    {
        public float[] Data;
        public int[] Shape;
        public int Rank;

        public bool RequiresGrad;
        public Tensor Gradient;
        public Func<Tensor, Tensor, Tensor, (Tensor, Tensor)> GradFn;
        public Tensor Father;
        public Tensor LeftLeaf;
        public Tensor RightLeaf;

        public static Tensor Zero = new Tensor([0f], [1], false);
        public static Tensor One = new Tensor([1f], [1], false);

        public Tensor(int[] shape, bool requiresGrad = false)
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

            Data = new float[shape.Aggregate((a, b) => a * b)];
            Shape = (int[])shape.Clone();
            Rank = shape.Length;

            RequiresGrad = requiresGrad;
            GradFn = null;
            Father = null;
            LeftLeaf = null;
            RightLeaf = null;
            if (requiresGrad)
            {
                Gradient = new Tensor(Shape, false);
            }
        }

        public Tensor(float[] data, int[] shape, bool requiresGrad = false)
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

            Data = (float[])data.Clone();
            Shape = (int[])shape.Clone();
            Rank = shape.Length;

            RequiresGrad = requiresGrad;
            GradFn = null;
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

        public static Tensor operator +(Tensor a, Tensor b) => TensorOperator.Add(a, b, true);
        public static Tensor operator -(Tensor a, Tensor b) => TensorOperator.Subtract(a, b, true);
        public static Tensor operator *(Tensor a, Tensor b) => TensorOperator.Multiply(a, b, true);
        public static Tensor operator /(Tensor a, Tensor b) => TensorOperator.Divide(a, b, true);
        public static Tensor operator -(Tensor a) => TensorOperator.Negate(a, true);

        public Tensor Repeat(int axis, int repeat = 2) => TensorOperator.Repeat(this, axis, repeat);
        public Tensor Reshape(int[] shape) => TensorOperator.Reshape(this, shape);

        public static Tensor Random(int[] shape, bool requiresGrad = false, float scale = 1f) => TensorOperator.Random(shape, requiresGrad, scale);
        public static Tensor Transpose(Tensor a) => TensorOperator.Transpose(a, true);
        public static Tensor MatMul(Tensor a, Tensor b) => TensorOperator.MatMul(a, b, true);

        public void Backward(Tensor externalGradient = null)
        {
            if (!RequiresGrad)
                throw new InvalidOperationException("Cannot call Backward on a tensor that does not require gradients.");

            if (externalGradient == null)
            {
                externalGradient = One;
            }

            Gradient = TensorOperator.Add(Gradient, externalGradient, false);

            if (GradFn != null)
            {
                (Tensor leftGrad, Tensor rightGrad) = GradFn(Gradient, LeftLeaf, RightLeaf);

                if (LeftLeaf != null && LeftLeaf.RequiresGrad)
                {
                    if (leftGrad == null)
                        throw new InvalidOperationException("Left gradient is null for a tensor that requires grad.");
                    LeftLeaf.Backward(leftGrad);
                }

                if (RightLeaf != null && RightLeaf.RequiresGrad)
                {
                    if (rightGrad == null)
                        throw new InvalidOperationException("Right gradient is null for a tensor that requires grad.");
                    RightLeaf.Backward(rightGrad);
                }
            }
        }
    }
}
