namespace VerbNet.Core
{
    public abstract class LossFunction
    {
        public float LossValue;
        public Tensor Loss;

        public abstract void Forward(Tensor pred, Tensor target);
        public abstract void Backward();
    }
}
