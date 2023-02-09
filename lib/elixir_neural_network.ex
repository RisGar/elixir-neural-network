defmodule ElixirNeuralNetwork do
  @moduledoc """
  Hauptmodul des Neuralen Netzwerks.
  """
  alias ElixirNeuralNetwork.Network

  def main do
    {training_data, _validation_data, _test_data} = Network.prepare_data(10)
    network = Network.init([784, 30, 10])
    Network.gradient_descent(network, training_data, 10, 3.0)
  end
end
