
using Microsoft.VisualBasic;

namespace VerbNet.Core
{
    public abstract class ActivationFunction : Layer
    {
        public override void ApplyGrad()
        {

        }

        public override void ZeroGrad()
        {

        }

        public override BinaryWriter Write(BinaryWriter bw)
        {
            return bw;
        }

        public override BinaryReader Read(BinaryReader br)
        {
            return br;
        }
    }
}
