namespace VerbNet.Core
{
    public class Tanh : ActivationFunction
    {
        public override Tensor Forward(Tensor input)
        {
            Tensor expX = Tensor.Exp(input);
            Tensor expNegX = Tensor.Exp(-input);

            return (expX - expNegX) / (expX + expNegX);
        }
    }
}
