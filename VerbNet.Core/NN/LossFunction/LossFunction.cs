using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
