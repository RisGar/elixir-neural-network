defmodule ElixirNeuralNetwork.Network do
  @moduledoc """
  Modul für die Kernlogik des neuralen Netzwerkes.
  """

  alias ElixirNeuralNetwork.Network

  defstruct num_layers: nil,
            sizes: nil,
            bias: nil,
            weights: nil

  @doc """
  Importiert den MNIST-Datensatz

      iex> download()
      {images, labels}
  """
  def download do
    Scidata.MNIST.download(cache_dir: "./cache")
  end

  def transform_images({binary, type, _shape}) do
    binary
    |> Nx.from_binary(type)
    |> Nx.reshape({60000, 784})
    |> Nx.divide(255)
  end

  def transform_labels({binary, type, _}) do
    binary
    |> Nx.from_binary(type)
    # vectorise result (convert from [0, 1, 2, 3, ...] to [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], ...])
    |> Nx.new_axis(-1)
    |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))
  end

  @doc """
  Bereitet den Datensatz zur Verarbeitung im neuralen Netz vor.
  Teilt die Daten in Trainings- Validations- und Testdaten ein.

      iex> transform_images({images, labels})
      {training_data, validation_data, test_data}
  """
  def prepare_data do
    {images, labels} = download()

    images =
      images
      |> transform_images()

    labels =
      labels
      |> transform_labels()

    for {image, label} <- images, labels do
      Nx.tensor(image, label)
    end

    # data = Enum.zip(images, labels)

    # training_count = floor(0.8 * Enum.count(data))
    # {training_data, test_data} = Enum.split(data, training_count)

    # validation_count = floor(0.2 * training_count)
    # {validation_data, training_data} = Enum.split(training_data, validation_count)

    # {training_data, validation_data, test_data}
  end

  @spec init(list) :: %ElixirNeuralNetwork.Network{
          bias: list,
          num_layers: non_neg_integer,
          sizes: list,
          weights: list
        }
  def init(sizes) do
    %Network{
      num_layers: length(sizes),
      sizes: sizes,
      bias:
        for y <- Enum.drop(sizes, 1) do
          random_normal({y, 1})
        end,
      weights:
        for {x, y} <- Enum.zip(Enum.drop(sizes, -1), Enum.drop(sizes, 1)) do
          random_normal({y, x})
        end
    }
  end

  defp random_normal({y, x}) do
    key = Nx.Random.key(:rand.uniform(255))
    Nx.Random.normal(key, shape: {y, x}) |> elem(0)
  end

  defp feedforward(b, w, a) do
    for {b, w} <- Enum.zip(b, w) do
      a = Nx.sigmoid(Nx.dot(w, a) + b)
    end

    a
  end

  def sgd() do
  end
end
