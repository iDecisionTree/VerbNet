namespace VerbNet.Core
{
    public abstract class Optimizer
    {
        public Tensor[] Parameters;
        public float LearningRate;

        public abstract void Step();
        public abstract void ZeroGrad();

        public Optimizer(Tensor[] parameters, float learningRate)
        {
            Parameters = parameters;
            LearningRate = learningRate;
        }
    }
}