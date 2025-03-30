using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace VerbNet.Core
{
    public class Linear : Layer
    {
        public Tensor Weight;
        public bool HasBias;
        public Tensor Bias;

        private int _inputSize;
        private int _outputSize;

        public Linear(int input, int output, bool hasBias = false, float learningRate = 0.001f)
        {
            _inputSize = input;
            _outputSize = output;
            LearningRate = learningRate;

            Weight = Tensor.Random([input, output], true, MathF.Sqrt(2f / (float)input));

            HasBias = hasBias;
            if (hasBias)
            {
                Bias = new Tensor([output], true);
            }
        }

        public override Tensor Forward(Tensor input)
        {
            if (input.Shape.Length != 2)
                throw new ArgumentException($"Input must be 2-dimensional tensor. Got shape: {string.Join(", ", input.Shape)}");
            if (input.Shape[1] != _inputSize)
                throw new ArgumentException($"Input feature dimension mismatch. Expected {_inputSize}, got {input.Shape[1]}");

            Tensor output;
            output = Tensor.MatMul(input, Weight);
            if (HasBias)
            {
                output += Bias;
            }

            return output;
        }

        public override void ApplyGrad()
        {
            for (int i = 0; i < Weight.Data.Length; i++)
            {
                Weight.Data[i] -= Weight.Gradient.Data[i] * LearningRate;
            }

            if(HasBias)
            {
                for (int i = 0; i < Bias.Data.Length; i++)
                {
                    Bias.Data[i] -= Bias.Gradient.Data[i] * LearningRate;
                }
            }
        }

        public override void ZeroGrad()
        {
            Array.Fill(Weight.Gradient.Data, 0f);
            if (HasBias)
            {
                Array.Fill(Bias.Gradient.Data, 0f);
            }
        }
    }
}
