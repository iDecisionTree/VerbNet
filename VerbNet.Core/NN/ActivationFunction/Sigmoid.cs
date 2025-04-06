namespace VerbNet.Core
{
    public class Sigmoid : ActivationFunction
    {
        public override Tensor Forward(Tensor input)
        {
            return Tensor.One / (Tensor.One + Tensor.Exp(-input));
        }
    }
}
