using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace VerbNet.Core
{
    public unsafe class AlignedArray<T> : IDisposable where T : unmanaged
    {
        public T* Ptr => _ptr;
        public Span<T> Span => AsSpan();
        public int Length => _length;
        public int Alignment => _alignment;

        private readonly T* _ptr;
        private readonly int _length;
        private readonly nuint _byteLength;
        private readonly int _alignment;
        private bool _disposed;

        public AlignedArray(int length, int alignment = 32)
        {
            if (length <= 0)
                throw new ArgumentOutOfRangeException(nameof(length), "Length must be positive");
            if (alignment <= 0 || (alignment & (alignment - 1)) != 0)
                throw new ArgumentException("Alignment must be a positive power of two", nameof(alignment));

            _length = length;
            _alignment = alignment;
            _byteLength = (nuint)(length * sizeof(T));

            _ptr = (T*)NativeMemory.AlignedAlloc(_byteLength, (nuint)alignment);
            if (_ptr == null)
                throw new OutOfMemoryException("Failed to allocate aligned memory");
            Clear();
        }

        public AlignedArray(T[] array, int alignment = 32) : this(array.Length, alignment)
        {
            array.AsSpan().CopyTo(AsSpan());
        }

        public ref T this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                if (_disposed)
                    throw new ObjectDisposedException(nameof(AlignedArray<T>));
                if ((uint)index >= (uint)_length)
                    throw new IndexOutOfRangeException($"Index {index} is out of range for length {_length}.");

                return ref _ptr[index];
            }
        }

        public Span<T> AsSpan()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(AlignedArray<T>));

            return new Span<T>(_ptr, _length);
        }

        public unsafe ReadOnlySpan<T> AsReadOnlySpan()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(AlignedArray<T>));

            return new ReadOnlySpan<T>(_ptr, _length);
        }

        public T[] ToArray()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(AlignedArray<T>));

            return AsSpan().ToArray();
        }

        public void Fill(T value)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(AlignedArray<T>));

            AsSpan().Fill(value);
        }

        public void Clear()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(AlignedArray<T>));

            AsSpan().Clear();
        }

        public AlignedArray<T> Clone()
        {
            AlignedArray<T> copy = new AlignedArray<T>(_length, _alignment);
            AsSpan().CopyTo(copy.AsSpan());

            return copy;
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            NativeMemory.AlignedFree(_ptr);
            _disposed = true;
            GC.SuppressFinalize(this);
        }

        ~AlignedArray()
        {
            Dispose();
        }
    }
}
