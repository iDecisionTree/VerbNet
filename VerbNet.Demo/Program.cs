using System.Diagnostics;
using VerbNet.Core;

namespace VerbNet.Demo
{
    internal class Program
    {
        static void Main(string[] args)
        {
            LayerList layers = new LayerList(
                new Linear(16, 128, true, 0.01f),
                new Linear(128, 128, true, 0.01f),
                new Linear(128, 16, true, 0.01f)
                );
            MSELoss mse = new MSELoss(true);

            Tensor input = Tensor.Random([4, 16]);
            Tensor target = Tensor.Random([4, 16]);

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
