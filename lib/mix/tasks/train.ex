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
  Die Daten werden in einer Matrix gespeichert, um dann mit einer Kostenfunktion trainiert zu werden,
  nachdem sie in das neurale Netz eingespeist werden. 
  Diese werden dann in die Trainings- und Testfunktion des neuralen Netzes eingespeist.
  Im Anschluss wird der trainierte Zustand in der Datei "state.axon" gespeichert.
  """
  @impl Mix.Task
  def run(_) do
    epochs = 10
    split = 0.8

    {images, labels} = download()

    {train_images, test_images} = batch_images(images, split)
    {train_labels, test_labels} = batch_labels(labels, split)

    {train_images, test_images}
    |> display_data()

    model = build()

    model
    |> display_network()

    state =
      model
      |> train(train_images, train_labels, epochs)

    model
    |> test(state, test_images, test_labels)

    save_state!(state, "cache/state.axon")

    :ok
  end
end
