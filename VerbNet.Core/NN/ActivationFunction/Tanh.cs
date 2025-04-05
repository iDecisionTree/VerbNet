using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
