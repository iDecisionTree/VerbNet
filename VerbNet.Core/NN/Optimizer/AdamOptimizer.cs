namespace VerbNet.Core
{
    public class AdamOptimizer : Optimizer
    {
        public float Beta1;
        public float Beta2;
        public float Epsilon;

        public int T;
        public Tensor[] M;
        public Tensor[] V;

        public AdamOptimizer(Tensor[] parameters, float learningRate, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 0.00000001f) : base(parameters, learningRate)
        {
            Beta1 = beta1;
            Beta2 = beta2;
            Epsilon = epsilon;

            T = 0;
            M = new Tensor[parameters.Length];
            V = new Tensor[parameters.Length];
            for (int i = 0; i < parameters.Length; i++)
            {
                M[i] = new Tensor(parameters[i].Shape, false);
                V[i] = new Tensor(parameters[i].Shape, false);
            }
        }

        public override void Step()
        {
            T++;
            for (int i = 0; i < Parameters.Length; i++)
            {
                M[i] = Beta1 * M[i] + (1f - Beta1) * Parameters[i].Gradient;
                V[i] = Beta2 * V[i] + (1f - Beta2) * TensorOperator.Power(Parameters[i].Gradient, 2f, false, false);

                Tensor mHat = M[i] / (1f - MathF.Pow(Beta1, T));
                Tensor vHat = V[i] / (1f - MathF.Pow(Beta2, T));

                Tensor delta = LearningRate * mHat / (TensorOperator.Sqrt(vHat, false, false) + Epsilon);

                // 不可以直接赋值，因为会破坏引用关系
                // Parameters[i] = TensorOperator.Subtract(Parameters[i], delta, false, true);

                Parallel.For(0, Parameters[i].Length, j =>
                {
                    Parameters[i].Data[j] -= delta.Data[j];
                });
            }
        }

        public override void ZeroGrad()
        {
            for (int i = 0; i < Parameters.Length; i++)
            {
                Parameters[i].Gradient.Data.Fill(0f);
            }
        }
    }
}