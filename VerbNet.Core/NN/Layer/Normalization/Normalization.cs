using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace VerbNet.Core
{
    public abstract class Normalization
    {
        public int InputSize;

        public Normalization(int input)
        {
            InputSize = input;
        }

        public abstract Tensor Forward(Tensor input);
    }
}
