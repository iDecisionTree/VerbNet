using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
