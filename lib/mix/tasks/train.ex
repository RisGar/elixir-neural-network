defmodule Mix.Tasks.Train do
  @moduledoc """
  Trainiert das neurale Netz mit dem MNIST-Datensatz.
  """

  use Mix.Task

  @requirements ["app.start", "app.config"]

  import ElixirNeuralNetwork

  @doc """
  Formatiert den MNIST-Datensatz und traniert das neurale Netz.

  Separiert den Datensatz in Trainings- und Testdaten.
  Diese werden dann in die Trainings- und Testfunktion des neuralen Netzes eingespeist.
  Im Anschluss wird das Modell und der Tranierte Zusatnd in der Datei "model.axon" gespeichert.
  """
  @impl Mix.Task
  def run(_) do
    # constants
    epochs = 5
    split = 0.8

    {images, labels} = download()

    {train_images, test_images} = transform_images(images, split)
    {train_labels, test_labels} = transform_labels(labels, split)

    {train_images, test_images}
    |> display_data()

    # 28x28 = 784 input neurons
    model = build({nil, 784})

    model
    |> display_network()

    model_state =
      model
      |> train(train_images, train_labels, epochs)

    model
    |> test(model_state, test_images, test_labels)

    {model, model_state}
    |> save!("model.axon")

    :ok
  end
end
