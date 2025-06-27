using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace VerbNet.Core
{
    public class GeLU : ActivationFunction
    {
        public readonly float Scale = MathF.Sqrt(2f / MathF.PI);
        public readonly float Alpha = 0.044715f;

        public override Tensor Forward(Tensor input)
        {
            Tensor xCubed = Tensor.Pow(input, 3f);
            Tensor inner = input + Tensor.Create(Alpha) * xCubed;

            Tensor tanhTerm = Tensor.Tanh(Tensor.Create(Scale) * inner);

            return Tensor.Create(0.5f) * input * (Tensor.One + tanhTerm);
        }
    }
}
