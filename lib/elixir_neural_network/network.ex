defmodule ElixirNeuralNetwork.Network do
  defstruct num_layers: nil,
            sizes: nil,
            bias: nil,
            weights: nil

  @spec init([non_neg_integer]) :: %ElixirNeuralNetwork.Network{
          num_layers: non_neg_integer,
          sizes: list,
          bias: list,
          weights: list
        }
  def init(sizes) do
    %ElixirNeuralNetwork.Network{
      num_layers: length(sizes),
      sizes: sizes,
      bias:
        for y <- Enum.drop(sizes, 1) do
          ElixirNeuralNetwork.Algebra.randn(y, 1)
        end,
      weights:
        for {x, y} <- Enum.zip(Enum.drop(sizes, -1), Enum.drop(sizes, 1)) do
          ElixirNeuralNetwork.Algebra.randn(y, x)
        end
    }
  end
end
