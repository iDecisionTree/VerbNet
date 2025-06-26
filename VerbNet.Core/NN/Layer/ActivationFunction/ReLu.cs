using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace VerbNet.Core
{
    public class ReLu : ActivationFunction
    {
        public override Tensor Forward(Tensor input)
        {
            return Tensor.Max(input, Tensor.Zero);
        }
    }
}
