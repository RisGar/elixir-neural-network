defmodule Mix.Tasks.Train do
  use Mix.Task

  @requirements ["app.start"]

  alias ElixirNeuralNetwork.Network

  def run(_) do
    {images, labels} = Network.download()

    {train_images, test_images} = Network.transform_images(images)
    {train_labels, test_labels} = Network.transform_labels(labels)

    model = Network.build({nil, 784}) |> IO.inspect()

    Network.display(model)

    model_state =
      model
      |> Network.train(train_images, train_labels, 5)

    model
    |> Network.test(model_state, test_images, test_labels)

    model
    |> Network.save!(model_state, "models/model.axon")
  end
end
