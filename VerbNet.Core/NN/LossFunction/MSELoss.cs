namespace VerbNet.Core
{
    public class MSELoss : LossFunction
    {
        public bool HasBatch;

        public MSELoss(bool hasBatch = true)
        {
            HasBatch = hasBatch;
        }

        public override void Forward(Tensor pred, Tensor target)
        {
            Tensor delta = pred - target;
            Loss = Tensor.Pow(delta, 2f);
            if (HasBatch)
            {
                Loss /= new Tensor([delta.Shape[0]], [1], false);
            }

            LossValue = 0f;
            for (int i = 0; i < Loss.Data.Length; i++)
            {
                LossValue += Loss.Data[i];
            }
            if (HasBatch)
            {
                LossValue /= delta.Shape[0];
            }
            else
            {
                LossValue /= delta.Data.Length;
            }
        }

        public override void Backward()
        {
            Loss.Backward();
        }
    }
}
