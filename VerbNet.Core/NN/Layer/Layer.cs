namespace VerbNet.Core
{
    public abstract class Layer
    {
        public float LearningRate;
        public string Name;

        public abstract Tensor Forward(Tensor input);
        public abstract void ApplyGrad();
        public abstract void ZeroGrad();
        public abstract Tensor[] GetParameters();
        public abstract BinaryWriter Write(BinaryWriter bw);
        public abstract BinaryReader Read(BinaryReader br);
    }
}
