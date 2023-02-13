defmodule ElixirNeuralNetwork do
  @moduledoc """
  Hauptmodul für die Kernlogik des neuralen Netzwerkes.
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

  defp training_split(data, split) do
    n = floor(split * Enum.count(data))
    Enum.split(data, n)
  end

  @doc """
  Wandelt die Bilder des Datensatz in für das Netz verarbeitbare Bilder zum erkennen einer Zahl um.
  """
  def prepare_images({binary, type, shape}) do
    binary
    |> Nx.from_binary(type)
    |> Nx.reshape({elem(shape, 0), 784})
    |> Nx.divide(255)
  end

  @doc """
  Wandelt die Bilder des Datensatz in für das Netz verarbeitbare "Batches" von Matrizen um.
  """
  def batch_images({binary, type, shape}, split) do
    binary
    |> Nx.from_binary(type)
    |> Nx.reshape({elem(shape, 0), 784})
    |> Nx.divide(255)
    |> Nx.to_batched(32)
    |> training_split(split)
  end

  @doc """
  Wandelt die gewünschten Ergebinisse des Datensatz in für das Netz verarbeitbare "Batches" von Matrizen um.
  """
  def batch_labels({binary, type, _}, split) do
    binary
    |> Nx.from_binary(type)
    # vectorise result (convert from [0, 1, 2, 3, ...] to [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], ...])
    |> Nx.new_axis(-1)
    |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))
    # batch result
    |> Nx.to_batched(32)
    |> training_split(split)
  end

  @doc """
  Stellt die Anzahl der Trainings- und Testdaten dar.
  """
  def display_data({training_data, testing_data}) do
    IO.puts("Training Images: #{training_data |> Enum.count()}")
    IO.puts("Testing Images: #{testing_data |> Enum.count()}")
  end

  @doc """
  Stellt das Netzwerk als Tebelle dar.
  """
  def display_network(model) do
    model
    |> Axon.Display.as_table(Nx.template({1, 784}, :f32))
    |> IO.puts()
  end

  @doc """
  Stellt das Bild als "Heatmap" dar. Hierdurch kann das Bild im Terminal dargestellt werden.
  """
  def display_image(image) do
    image
    |> Nx.reshape({28, 28})
    |> Nx.to_heatmap()
    |> IO.inspect()
  end

  @doc ~S"""
  Erstellt ein neurales Netz.

  Das Netz hat 28x28 Input-Neuronen, 128 "versteckte" Neuronen und 10 Ergebnis-Neuronen.
  Als Aktivierungs-Funktion wird hier für beide Ebenen die Logistische Funktion $\sigma(x)$ verwendet.
  """
  def build() do
    Axon.input("input", shape: {nil, 784})
    |> Axon.dense(128, name: "hidden", activation: :relu)
    |> Axon.dropout(name: "dropout", rate: 0.5)
    |> Axon.dense(10, name: "output", activation: :sigmoid)
  end

  @doc """
  Trainiert das neurale Netz.

  Hierzu wird es über mehrere Generationen "Epochen" mit dem Trainingsdatensatz trainiert.
  """
  def train(model, train_images, train_labels, epochs) do
    model
    |> Axon.Loop.trainer(:categorical_cross_entropy, :adamw)
    |> Axon.Loop.metric(:accuracy, "Accuracy")
    |> Axon.Loop.run(Stream.zip(train_images, train_labels), %{}, epochs: epochs, compiler: EXLA)
  end

  @doc """
  Testet das neurale Netz.

  Hierzu werden über eine Generation die Testdaten in das Netz eingeführt und die Genauigkeit des Netzes erfasst.
  """
  def test(model, model_state, test_images, test_labels) do
    model
    |> Axon.Loop.evaluator()
    |> Axon.Loop.metric(:accuracy, "Accuracy")
    |> Axon.Loop.run(Stream.zip(test_images, test_labels), model_state, compiler: EXLA)
  end

  @doc """
  Führt dem Netzwerk ein Bild ein, um die Ziffer zu erhalelten, welche das Netzwerk als höchste Warscheinlichkeit sieht.
  """
  def predict(model, state, data) do
    model
    |> Axon.predict(state, data)
    |> Nx.argmax()
  end

  @doc """
  Speichert das Netzwerk und dessen Zusaznd in der angegebenen Datei.
  """
  def save_state!(state, path) do
    File.rm(path)
    File.write!(path, Nx.serialize(state))
  end

  @doc """
  Lädt das Netzwerk und dessen Zustand aus der angegebenen Datei.
  """
  def load_state!(path) do
    path
    |> File.read!()
    |> Nx.deserialize()
  end
end
