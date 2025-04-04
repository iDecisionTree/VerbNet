namespace VerbNet.Core
{
    public abstract class Layer
    {
        public float LearningRate;

        public abstract Tensor Forward(Tensor input);
        public abstract void ApplyGrad();
        public abstract void ZeroGrad();
    }
}
