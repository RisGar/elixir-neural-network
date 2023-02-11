defmodule Mix.Tasks.Predict do
  use Mix.Task

  @requirements ["app.start", "app.config"]

  import ElixirNeuralNetwork

  @impl Mix.Task
  def run(_) do
    {images, _} = download()

    prediction_images = prediction_images(images)

    model = build({nil, 784})
    state = load_state!("cache/state.axon")

    display_network(model)

    sample =
      prediction_images
      |> Enum.at(0)

    display_image(sample)

    predict({model, state}, sample)

    :ok
  end
end
