namespace VerbNet.Core
{
    public class AdamOptimizer : Optimizer
    {
        public float Beta1;
        public float Beta2;
        public float Epsilon;

        public int T;
        public float[][] M;
        public float[][] V;

        public AdamOptimizer(Tensor[] parameters, float learningRate, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 0.00000001f) : base(parameters, learningRate)
        {
            Beta1 = beta1;
            Beta2 = beta2;
            Epsilon = epsilon;

            T = 0;
            M = new float[parameters.Length][];
            V = new float[parameters.Length][];
            for (int i = 0; i < parameters.Length; i++)
            {
                M[i] = new float[parameters[i].Length];
                V[i] = new float[parameters[i].Length];
            }
        }

        public override void Step()
        {
            T++;
            for (int i = 0; i < Parameters.Length; i++)
            {
                for (int j = 0; j < Parameters[i].Length; j++)
                {
                    M[i][j] = Beta1 * M[i][j] + (1f - Beta1) * Parameters[i].Gradient.Data[j];
                    V[i][j] = Beta2 * V[i][j] + (1f - Beta2) * MathF.Pow(Parameters[i].Gradient.Data[j], 2f);
                    float mHat = M[i][j] / (1f - MathF.Pow(Beta1, T));
                    float vHat = V[i][j] / (1f - MathF.Pow(Beta2, T));

                    Parameters[i].Data[j] -= LearningRate * mHat / (MathF.Sqrt(vHat) + Epsilon);
                }
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