using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace VerbNet.Core
{
    public abstract class Normalization : Layer
    {
        public int InputSize;

        public Normalization(int input)
        {
            InputSize = input;
        }

        public override void ApplyGrad()
        {
            
        }

        public override BinaryReader Read(BinaryReader br)
        {
            return br;
        }

        public override BinaryWriter Write(BinaryWriter bw)
        {
            return bw;
        }

        public override Tensor[] GetParameters()
        {
            return new Tensor[0];
        }

        public override void ZeroGrad()
        {
            
        }
    }
}
