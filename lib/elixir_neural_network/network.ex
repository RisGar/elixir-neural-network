defmodule ElixirNeuralNetwork.Network do
  @moduledoc """
  Modul für die Kernlogik des neuralen Netzwerkes.
  """

  @doc """
  Importiert den MNIST-Datensatz

  Der MNIST-Datensatz besteht aus 60.000 28x28 Pixel großen Bildern von handgeschriebenen Zahlen.
  (https://en.wikipedia.org/wiki/MNIST_database)

  Die Daten werden in einem "Tuple", ein Container für eine bestimmte Anzahl von Elementen, gespeichert.

  ## Format

      iex> download()
      {images, labels}
  """
  def download do
    Scidata.MNIST.download(cache_dir: "./cache")
  end

  def transform_images({binary, type, shape}) do
    binary
    |> Nx.from_binary(type)
    |> Nx.reshape({elem(shape, 0), 784})
    |> Nx.divide(255)
    |> Nx.to_batched(32)
    |> Enum.split(1750)
  end

  def transform_labels({binary, type, _}) do
    binary
    |> Nx.from_binary(type)
    # vectorise result (convert from [0, 1, 2, 3, ...] to [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], ...])
    |> Nx.new_axis(-1)
    |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))
    |> Nx.to_batched(32)
    |> Enum.split(1750)
  end

  @doc ~S"""
  Erstellt ein neurales Netz mit 28x28 Input-Neuronen, 128 "versteckte" Neuronen und 10 Ergebnis-Neuronen.

  Als Aktivierungs-Funktion wird hier für beide Ebenen die Logistische Funktion $\sigma(x)$ verwendet.
  """
  def build(input_shape) do
    Axon.input("input", shape: input_shape)
    |> Axon.dense(128, activation: :sigmoid)
    |> Axon.dropout()
    |> Axon.dense(10, activation: :sigmoid)
  end

  def train(model, train_images, train_labels, epochs) do
    model
    |> Axon.Loop.trainer(:categorical_cross_entropy, Axon.Optimizers.sgd(0.01))
    |> Axon.Loop.metric(:accuracy, "Accuracy")
    |> Axon.Loop.run(Stream.zip(train_images, train_labels), %{}, epochs: epochs)
  end

  def test(model, model_state, test_images, test_labels) do
    model
    |> Axon.Loop.evaluator()
    |> Axon.Loop.metric(:accuracy, "Accuracy")
    |> Axon.Loop.run(Stream.zip(test_images, test_labels), model_state)
  end

  def display(model) do
    model
    |> Axon.Display.as_table(Nx.template({1, 784}, :f32))
    |> IO.puts()
  end

  def save!(model, state, path) do
    contents = :erlang.term_to_binary({model, state})

    File.write!(path, contents)
  end

  def load!(path) do
    path
    |> File.read!()
    |> :erlang.binary_to_term()
  end
end
