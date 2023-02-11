defmodule Mix.Tasks.Predict do
  use Mix.Task

  @requirements ["app.start", "app.config"]

  import ElixirNeuralNetwork

  @impl Mix.Task
  def run(_) do
    {data, _} = download()

    {images, _} = transform_images(data, 1)

    # image = images > Nx.slice_along_axis(0, 1)

    {model, model_state} = load!("model.axon")

    display_network(model)
    # display_image(image)

    samples =
      images
      |> Enum.at(0)

    predict({model, model_state}, samples)

    :ok
  end
end
