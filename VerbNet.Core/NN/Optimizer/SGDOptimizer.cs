namespace VerbNet.Core
{
    public class SGDOptimizer : Optimizer
    {
        public SGDOptimizer(Tensor[] parameters, float learningRate) : base(parameters, learningRate)
        {

        }

        public override void Step()
        {
            for (int i = 0; i < Parameters.Length; i++)
            {
                Tensor delta = LearningRate * Parameters[i].Gradient;
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