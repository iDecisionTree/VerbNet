using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace VerbNet.Core
{
    public class L1Loss : LossFunction
    {
        public override void Forward(Tensor pred, Tensor target)
        {
            Tensor delta = pred - target;
            Loss = Tensor.Abs(delta);

            LossValue = 0f;
            for (int i = 0; i < Loss.Data.Length; i++)
            {
                LossValue += Loss.Data[i];
            }
            LossValue /= Loss.Data.Length;
        }

        public override void Backward()
        {
            Loss.Backward();
        }
    }
}
