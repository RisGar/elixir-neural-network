defmodule Mix.Tasks.Predict do
  use Mix.Task

  @requirements ["app.start", "app.config"]

  import ElixirNeuralNetwork

  @impl Mix.Task
  def run(_) do
    {images, _} = download()

    images = prediction_images(images)

    image =
      images
      |> Nx.slice_along_axis(0, 1)

    {model, model_state} = load!("model.axon")

    display_network(model)
    display_image(image)

    sample =
      image
      |> Nx.reshape({1, 784})
      |> List.wrap()
      |> Nx.stack()

    predict({model, model_state}, sample)

    :ok
  end
end
