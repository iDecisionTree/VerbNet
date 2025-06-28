using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace VerbNet.Core
{
    public class LayerNorm : Normalization
    {
        public const float Epsilon = 0.0001f;

        public LayerNorm(int input) : base(input)
        {
        }

        public override Tensor Forward(Tensor input)
        {
            Tensor mean = Tensor.Mean(input, -1, true);
            Tensor variance = Tensor.Variance(input, -1, true);
            Tensor sqrt = Tensor.Sqrt(variance + Tensor.Create(Epsilon));

            return (input - mean) / sqrt;
        }
    }
}
