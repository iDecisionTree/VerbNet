using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
    }
}
