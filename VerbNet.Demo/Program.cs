using System.Diagnostics;
using System.Net.Http.Headers;
using VerbNet.Core;

namespace VerbNet.Demo
{
    internal class Program
    {
        static void Main(string[] args)
        {
            
            LayerList layers = new LayerList(
                new Linear(64, 1024, true, 0.001f, "linear1"),
                new Linear(1024, 1024, true, 0.0001f, "linear2"),
                new Linear(1024, 1, true, 0.001f, "linear3"),
                new Tanh()
                );
            MSELoss mse = new MSELoss();
            SGDOptimizer optim = new SGDOptimizer(layers.GetParameters(), 0.0001f);

            Tensor input = Tensor.Random([4, 64]);
            Tensor target = Tensor.Random([4, 1]);

            Stopwatch stopwatch = new Stopwatch();

            float[] times = new float[50];
            for (int i = 0; i < times.Length; i++)
            {
                optim.ZeroGrad();

                stopwatch.Restart();

                Tensor output = layers.Forward(input);
                mse.Forward(output, target);
                mse.Backward();

                stopwatch.Stop();

                times[i] = stopwatch.ElapsedMilliseconds;
                Console.WriteLine($"Epoch: {i}/{times.Length}, Loss: {mse.LossValue}, Time: {stopwatch.ElapsedMilliseconds}ms");

                optim.Step();
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