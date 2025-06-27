namespace VerbNet.Core
{
    public class Linear : Layer
    {
        public Tensor Weight;
        public bool HasBias;
        public Tensor Bias;

        public int InputSize;
        public int OutputSize;

        public Linear(int input, int output, bool hasBias = false, float learningRate = 0.001f, string name = "")
        {
            InputSize = input;
            OutputSize = output;
            LearningRate = learningRate;
            Name = name;

            Weight = Tensor.Random([input, output], MathF.Sqrt(2f / (float)input), true, $"{Name}_weight");

            HasBias = hasBias;
            if (hasBias)
            {
                Bias = new Tensor([output], true, $"{Name}_bias");
            }
        }

        public override Tensor Forward(Tensor input)
        {
            if (input.Shape.Length != 2)
                throw new ArgumentException($"Input must be 2-dimensional tensor. Got shape: {string.Join(", ", input.Shape)}");
            if (input.Shape[1] != InputSize)
                throw new ArgumentException($"Input feature dimension mismatch. Expected {InputSize}, got {input.Shape[1]}");

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

            if (HasBias)
            {
                for (int i = 0; i < Bias.Data.Length; i++)
                {
                    Bias.Data[i] -= Bias.Gradient.Data[i] * LearningRate;
                }
            }
        }

        public override void ZeroGrad()
        {
            Weight.Gradient.Data.Fill(0f);
            if (HasBias)
            {
                Bias.Gradient.Data.Fill(0f);
            }
        }

        public override Tensor[] GetParameters()
        {
            List<Tensor> parameters = new List<Tensor>();
            parameters.Add(Weight);
            if (HasBias)
            {
                parameters.Add(Bias);
            }

            return parameters.ToArray();
        }

        public override BinaryWriter Write(BinaryWriter bw)
        {
            bw.Write((int)LayerType.Linear);
            bw.Write(Name);
            bw.Write(LearningRate);
            bw.Write(InputSize);
            bw.Write(OutputSize);
            bw = Weight.Write(bw);
            bw.Write(HasBias);
            if (HasBias)
            {
                bw = Bias.Write(bw);
            }

            return bw;
        }

        public override BinaryReader Read(BinaryReader br)
        {
            if (br.ReadInt32() == (int)LayerType.Linear)
            {
                Name = br.ReadString();
                LearningRate = br.ReadSingle();
                InputSize = br.ReadInt32();
                OutputSize = br.ReadInt32();
                br = Weight.Read(br);
                HasBias = br.ReadBoolean();
                if (HasBias)
                {
                    br = Bias.Read(br);
                }
            }
            else
            {
                throw new InvalidDataException("Invalid layer type.");
            }

            return br;
        }
    }
}
