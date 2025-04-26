using System.Reflection.Metadata.Ecma335;

namespace VerbNet.Core
{
    public class LayerList
    {
        public List<Layer> Layers = new List<Layer>();

        public LayerList(params Layer[] layers)
        {
            Layers = layers.ToList();
        }

        public void Add(Layer layer)
        {
            Layers.Add(layer);
        }

        public Tensor Forward(Tensor input)
        {
            Tensor output = input;
            for (int i = 0; i < Layers.Count; i++)
            {
                output = Layers[i].Forward(output);
            }

            return output;
        }

        public void ApplyGrad()
        {
            for (int i = 0; i < Layers.Count; i++)
            {
                Layers[i].ApplyGrad();
            }
        }

        public void ZeroGrad()
        {
            for (int i = 0; i < Layers.Count; i++)
            {
                Layers[i].ZeroGrad();
            }
        }

        public Tensor[] GetParameters()
        {
            List<Tensor> parameters = new List<Tensor>();
            for (int i = 0; i < Layers.Count; i++)
            {
                Tensor[] layerParams = Layers[i].GetParameters();
                for (int j = 0; j < layerParams.Length; j++)
                {
                    parameters.Add(layerParams[j]);
                }
            }

            return parameters.ToArray();
        }

        public void Save(string path)
        {
            using (FileStream fs = new FileStream(path, FileMode.OpenOrCreate))
            {
                using (BinaryWriter bw = new BinaryWriter(fs))
                {
                    for (int i = 0; i < Layers.Count; i++)
                    {
                        Layers[i].Write(bw);
                    }    
                }
            }
        }

        public void Load(string path)
        {
            using (FileStream fs = new FileStream(path, FileMode.Open))
            {
                using (BinaryReader br = new BinaryReader(fs))
                {
                    for (int i = 0; i < Layers.Count; i++)
                    {
                        Layers[i].Read(br);
                    }
                }   
            }
        }
    }
}
