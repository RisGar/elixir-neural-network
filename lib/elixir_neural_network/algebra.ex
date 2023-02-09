defmodule ElixirNeuralNetwork.Algebra do
  @moduledoc """
  Modul für wiederverwendete algebraische Funktionen.
  Diese enthalten u. a. die Aktivierungsfunktion.
  """

  @doc ~S"""
  Logistische Funktion $\sigma(x)$, eine Funktion mit einem S-förmigen Graphen.

  Im wesentlichen nur eine skalierte & verschobene hyperbolische Tangensfunktion.
  https://de.wikipedia.org/wiki/Logistische_Funktion

  Wird für dieses Netz als Aktivierungsfunktion verwendet.
  """
  @spec logistic(number) :: float
  def logistic(x) do
    1.0 / (1.0 + :math.exp(-x))
  end

  @doc ~S"""
  Implementation der numpy-Funktion `randn()` für 2 Dimensionen.

  Generieret zufällige Werte einer Normalverteilung mit einer
  Standardabweichung / Varianz $\sigma = 1$ und einem Erwartungswert $\mu = 0$.
  Erstellt eine 2-Dimensionale Matrix, bei der jede Reihe durch eine Liste dargestellt wird.
  https://de.wikipedia.org/wiki/Normalverteilung
  """
  @spec randn(integer, integer) :: [[number]]
  def randn(y, x) do
    for _ <- 1..y do
      for _ <- 1..x do
        :rand.normal()
      end
    end
  end
end
