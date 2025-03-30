using VerbNet.Core;

namespace VerbNet.Demo
{
    internal class Program
    {
        static void Main(string[] args)
        {
            LayerList layers = new LayerList(
                new Linear(16, 512, true, 0.001f),
                new Linear(512, 512, true, 0.001f),
                new Linear(512, 16, true, 0.001f)
                );
            MSELoss mse = new MSELoss(true);

            Tensor input = Tensor.Random([64, 16]);
            Tensor target = Tensor.Random([64, 16]);

            for (int i = 0; i < 1000000; i++)
            {
                layers.ZeroGrad();

                Tensor output = layers.Forward(input);
                mse.Forward(output, target);
                mse.Backward();

                Console.WriteLine(mse.LossValue);

                layers.ApplyGrad();
            }    
        }
    }
}
