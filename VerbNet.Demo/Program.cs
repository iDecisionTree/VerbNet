using System.Diagnostics;
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

            Stopwatch stopwatch = new Stopwatch();

            for (int i = 0; i < 1000000; i++)
            {
                layers.ZeroGrad();

                stopwatch.Restart();

                Tensor output = layers.Forward(input);
                mse.Forward(output, target);
                mse.Backward();

                stopwatch.Stop();

                Console.WriteLine($"Epoch: {i}/{1000000}, Loss: {mse.LossValue}, Time: {stopwatch.ElapsedMilliseconds}ms");

                layers.ApplyGrad();
            }
        }
    }
}
