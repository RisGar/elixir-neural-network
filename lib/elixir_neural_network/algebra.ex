defmodule ElixirNeuralNetwork.Algebra do
  @moduledoc """
  Modul für wiederverwendete algebraische Funktionen.
  Diese enthalten u. a. die Aktivierungsfunktion.
  """

  @doc """
  Logistische Funktion, eine Funktion mit einem "S"-förmigen Graphen.
  Im wesentlichen nur eine skalierte & verschobene hyperbolische Tangensfunktion.
  https://de.wikipedia.org/wiki/Logistische_Funktion

  Wird für dieses Netz als Aktivierungsfunktion verwendet.
  """
  @spec logistic(number) :: float
  def logistic (x) do
    1.0 / (1.0+:math.exp(-x))
  end

  @doc """
  Benutzen der Reziprokenregel zum Ableiten von sigma(x)
  https://de.wikipedia.org/wiki/Reziprokenregel
  """
  @spec logistic_prime(number) :: float
  def logistic_prime (x) do
    logistic(x) * (1-logistic(x))
  end
end
