defmodule ElixirNeuralNetwork do
  @moduledoc """
  Hauptmodul des Neuralen Netzwerks.
  """
  alias ElixirNeuralNetwork.Network

  def main do
    Network.init([784, 30, 10])
  end
end
