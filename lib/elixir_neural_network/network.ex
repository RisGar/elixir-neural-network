defmodule ElixirNeuralNetwork.Network do
  @moduledoc """
  Modul für die Kernlogik des neuralen Netzwerkes.
  """

  import Nx.Defn

  alias ElixirNeuralNetwork.Network

  defstruct num_layers: nil,
            sizes: nil,
            biases: nil,
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
  @spec prepare_data(pos_integer) ::
          {[{Nx.Tensor, Nx.Tensor}], [{Nx.Tensor, Nx.Tensor}], [{Nx.Tensor, Nx.Tensor}]}
  def prepare_data(batch_size) do
    {images, labels} = download()

    images =
      images
      |> transform_images()
      |> Nx.to_batched(batch_size)

    labels =
      labels
      |> transform_labels()
      |> Nx.to_batched(batch_size)

    data = Enum.zip(images, labels)

    training_count = floor(0.8 * Enum.count(data))
    validation_count = floor(0.2 * training_count)

    {training_data, test_data} = Enum.split(data, training_count)
    {validation_data, training_data} = Enum.split(training_data, validation_count)

    {training_data, validation_data, test_data}
  end

  defp random_normal({y, x}) do
    key = Nx.Random.key(:rand.uniform(255))
    Nx.Random.normal(key, shape: {y, x}) |> elem(0)
  end

  @doc """
  Initiert die Werte des neuralen Netzwerks.

  Als Parameter nimmt die Funktion die Dimensionen, die das Netz annehmen soll.

  ## Beispiel

      iex> init_network([2, 4, 2])
      %Network{num_layers: 3, sizes: [2, 4, 2], bias: [...], weights: [...]}
  """
  @spec init(list) :: %ElixirNeuralNetwork.Network{
          biases: list,
          num_layers: non_neg_integer,
          sizes: list,
          weights: list
        }
  def init(sizes) do
    %Network{
      num_layers: length(sizes),
      sizes: sizes,
      biases:
        for y <- Enum.drop(sizes, 1) do
          random_normal({y, 1})
        end,
      weights:
        for {x, y} <- Enum.zip(Enum.drop(sizes, -1), Enum.drop(sizes, 1)) do
          random_normal({y, x})
        end
    }
  end

  defp back_propagation do
  end

  defp update_batch(network, batch, learning_rate) do
    nabla_b = Nx.tensor()
  end

  def gradient_descent(network, training_data, epochs, learning_rate) do
    for j <- 1..epochs do
      for batch <- training_data do
        update_batch(network, batch, learning_rate)
      end

      IO.puts("Epoch: #{j} completed")
    end
  end

  def new() do
    Axon.input("input", shape: {nil, 784})
    |> Axon.dense(128, activation: :sigmoid)
    |> Axon.dense(10, activation: :sigmoid)
    |> Axon.Display.as_table(Nx.template({1, 784}, :f32))
    |> IO.puts()
  end

  def train(model, training_data, validation_data) do
    model
    |> Axon.Loop.trainer(:categorical_cross_entropy, Axon.Optimizers.sgd(0.01))
    |> Axon.Loop.validate(model, validation_data)
    |> Axon.Loop.run(training_data, epochs: 10, compiler: EXLA)
  end

  def test(model, test_data) do
    model
    |> Axon.Loop.evaluator()
    |> Axon.Loop.run(test_data, compiler: EXLA)
  end

  def save!(model, state) do
    contents = :erlang.term_to_binary({model, state})

    File.write!(path(), contents)
  end

  def load! do
    path()
    |> File.read!()
    |> :erlang.binary_to_term()
  end

  def path do
    Path.join(Application.app_dir(:digits, "priv"), "model.axon")
  end
end
