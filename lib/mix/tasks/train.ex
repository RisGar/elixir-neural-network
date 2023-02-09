defmodule Mix.Tasks.Train do
  use Mix.Task

  @requirements ["app.start"]

  alias ElixirNeuralNetwork.Network

  def run(_) do
    {training_data, validation_data, test_data} = Network.prepare_data(32)

    model = Network.new()
    Network.train(model, training_data, validation_data)
  end
end
