using System.Diagnostics;
using VerbNet.Core;

namespace VerbNet.Demo
{
    internal class Program
    {
        static void Main(string[] args)
        {
            LayerList layers = new LayerList(
                new Linear(16, 1024, true, 0.001f, "linear1"),
                new ReLU(),
                new Linear(1024, 1024, true, 0.0001f, "linear2"),
                new ReLU(),
                new Linear(1024, 1, true, 0.001f, "linear3")
                );
            MSELoss mse = new MSELoss();
            AdamOptimizer optim = new AdamOptimizer(layers.GetParameters(), 0.0001f);

            Tensor input = Tensor.Random([1, 16]);
            Tensor target = Tensor.Random([1, 1]);

            Stopwatch stopwatch = new Stopwatch();

            float[] times = new float[2000];
            for (int i = 0; i < times.Length; i++)
            {
                optim.ZeroGrad();

                stopwatch.Restart();

                Tensor output = layers.Forward(input);
                mse.Forward(output, target);
                mse.Backward();

                optim.Step();

                stopwatch.Stop(); 
                times[i] = stopwatch.ElapsedMilliseconds;
                Console.WriteLine($"Epoch: {i}/{times.Length}, Loss: {mse.LossValue}, Time: {stopwatch.ElapsedMilliseconds}ms");
            }

            float avgTime = 0f;
            for (int i = 0; i < times.Length; i++)
            {
                avgTime += times[i];
            }
            avgTime /= times.Length;
            Console.WriteLine($"Average Time: {avgTime}ms");

            Console.ReadLine();

            layers.Save("TestModel.bin");

            //layers.Load("TestModel.bin");
        }
    }
}