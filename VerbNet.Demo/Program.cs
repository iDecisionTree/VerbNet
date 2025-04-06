using System.Diagnostics;
using VerbNet.Core;

namespace VerbNet.Demo
{
    internal class Program
    {
        static void Main(string[] args)
        {
            LayerList layers = new LayerList(
                new Linear(64, 1024, true, 0.001f),
                new Linear(1024, 1024, true, 0.0001f),
                new Linear(1024, 1, true, 0.001f),
                new Tanh()
                );
            L1Loss l1 = new L1Loss();

            Tensor input = Tensor.Random([4, 64]);
            Tensor target = Tensor.Random([4, 1]);

            Stopwatch stopwatch = new Stopwatch();

            float[] times = new float[500];
            for (int i = 0; i < times.Length; i++)
            {
                layers.ZeroGrad();

                stopwatch.Restart();

                Tensor output = layers.Forward(input);
                l1.Forward(output, target);
                l1.Backward();

                stopwatch.Stop();

                times[i] = stopwatch.ElapsedMilliseconds;
                Console.WriteLine($"Epoch: {i}/{times.Length}, Loss: {l1.LossValue}, Time: {stopwatch.ElapsedMilliseconds}ms");

                layers.ApplyGrad();
            }

            float avgTime = 0f;
            for (int i = 0; i < times.Length; i++)
            {
                avgTime += times[i];
            }
            avgTime /= times.Length;
            Console.WriteLine($"Average Time: {avgTime}ms");
        }
    }
}
