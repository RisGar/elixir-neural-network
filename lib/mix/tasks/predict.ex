defmodule Mix.Tasks.Predict do
  use Mix.Task

  @requirements ["app.start", "app.config"]

  import ElixirNeuralNetwork

  @impl Mix.Task
  def run(_) do
    {data, _} = download()

    {images, _} = transform_images(data, 1)

    {model, state} = load!("model.axon")

    display_network(model)

    samples =
      images
      |> Enum.at(0)

    display_image(samples |> Nx.slice_along_axis(0, 1))

    predict({model, state}, samples)

    :ok
  end
end
